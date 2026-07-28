import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
from pathlib import Path

from augment import TimeSeriesAugmenter
from loss import ContrastiveClusteringLoss

class InstanceNormalization1D(nn.Module):
    def __init__(self, eps=1e-5):
        super(InstanceNormalization1D, self).__init__()
        self.eps = eps

    def forward(self, x):
        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.std(x, dim=1, keepdim=True, unbiased=False)
        # Prevent dividing by 0 or amplifying tiny noise
        std = torch.where(std > 1e-4, std, torch.ones_like(std))
        return (x - mean) / std

class VariableSelectionNetwork(nn.Module):
    def __init__(self, num_features, hidden_size, dropout=0.2):
        super(VariableSelectionNetwork, self).__init__()
        self.weight_network = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_features),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        weights = self.weight_network(x)
        return x * weights * x.shape[-1]

class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_over_clusters=15, dropout=0.2):
        super(TemporalFusionTransformer, self).__init__()
        self.norm = InstanceNormalization1D()
        self.vsn = VariableSelectionNetwork(input_size, hidden_size, dropout)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        self.attn_layer_norm = nn.LayerNorm(hidden_size)
        
        # Instance Head for InfoNCE
        self.instance_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 32)
        )
        
        # Over-Cluster Head
        self.num_over_clusters = num_over_clusters
        self.over_cluster_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, num_over_clusters),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.vsn(x)
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Residual connection and LayerNorm
        attn_out = self.attn_layer_norm(lstm_out + attn_out)
        
        # Backbone embedding h
        h = attn_out[:, -1, :]
        
        z = self.instance_head(h)
        c_over = self.over_cluster_head(h)
        
        # Merge over-clusters (e.g. 15) to macro-clusters (3)
        # Sum every 5 probabilities
        group_size = self.num_over_clusters // 3
        c_macro = c_over.view(-1, 3, group_size).sum(dim=2)
        
        return h, z, c_over, c_macro

class PricePredictor:
    def __init__(self, input_chunk_length=30, hidden_size=64, num_layers=2):
        self.input_chunk_length = input_chunk_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if not torch.cuda.is_available():
            raise RuntimeError("GPU requested but CUDA is not available!")
        self.device = torch.device('cuda')
        self.model = None
        self.history = {'train_loss': [], 'val_loss': []}
        self.augmenter = TimeSeriesAugmenter(self.device)

    def create_sequences(self, features_array, target):
        X, y = [], []
        for i in range(len(features_array) - self.input_chunk_length):
            X.append(features_array[i:(i + self.input_chunk_length)])
            y.append(target[i + self.input_chunk_length])
        return np.array(X), np.array(y)

    def train(self, train_csv, val_csv, epochs=50, batch_size=256):
        train_df = pd.read_csv(train_csv, index_col=0, parse_dates=True)
        val_df = pd.read_csv(val_csv, index_col=0, parse_dates=True)

        features_train = train_df.drop(columns=['target']).values
        target_train = train_df['target'].values
        features_val = val_df.drop(columns=['target']).values
        target_val = val_df['target'].values

        X_train, y_train = self.create_sequences(features_train, target_train)
        X_val, y_val = self.create_sequences(features_val, target_val)

        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.LongTensor(y_val).to(self.device)

        input_size = X_train.shape[2]
        self.model = TemporalFusionTransformer(
            input_size=input_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers
        ).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        # Note: criterion size must exactly match batch_size during training.
        # We handle remaining batches by dropping them.
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        criterion = ContrastiveClusteringLoss(batch_size=batch_size)

        print("Starting Self-Supervised Contrastive Training...")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch_X, _ in train_loader:
                optimizer.zero_grad()
                
                # Create two augmented views
                x_i = self.augmenter.augment(batch_X)
                x_j = self.augmenter.augment(batch_X)
                
                # Forward pass both views
                _, z_i, c_i, _ = self.model(x_i)
                _, z_j, c_j, _ = self.model(x_j)
                
                # CC Loss
                loss = criterion(z_i, z_j, c_i, c_j)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            avg_train_loss = total_loss / len(train_loader)
            self.history['train_loss'].append(avg_train_loss)
            
            # Validation
            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch_X, _ in val_loader:
                    x_i = self.augmenter.augment(batch_X)
                    x_j = self.augmenter.augment(batch_X)
                    _, z_i, c_i, _ = self.model(x_i)
                    _, z_j, c_j, _ = self.model(x_j)
                    loss = criterion(z_i, z_j, c_i, c_j)
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            self.history['val_loss'].append(avg_val_loss)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        return self.history['train_loss'][-1], self.history['val_loss'][-1]

    def predict(self, data):
        self.model.eval()
        if isinstance(data, pd.DataFrame):
            features = data.drop(columns=['target'], errors='ignore').values
        else:
            features = data
            
        X = []
        if len(features) >= self.input_chunk_length:
            X.append(features[-self.input_chunk_length:])
        else:
            return 0.0, 0.0, 0.0
            
        X = torch.FloatTensor(np.array(X)).to(self.device)
        with torch.no_grad():
            _, _, _, c_macro = self.model(X)
            p_flat, p_up, p_down = c_macro[0].cpu().numpy()
        return p_flat, p_up, p_down

    def predict_batch(self, data, batch_size=512):
        self.model.eval()
        if isinstance(data, pd.DataFrame):
            features = data.drop(columns=['target'], errors='ignore').values
        else:
            features = data
            
        X = []
        for i in range(len(features) - self.input_chunk_length + 1):
            X.append(features[i:(i + self.input_chunk_length)])
            
        if not X:
            return np.array([])
            
        X = torch.FloatTensor(np.array(X)).to(self.device)
        dataset = torch.utils.data.TensorDataset(X)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_probs = []
        with torch.no_grad():
            for batch_X, in loader:
                _, _, _, c_macro = self.model(batch_X)
                all_probs.append(c_macro.cpu().numpy())
                
        return np.concatenate(all_probs, axis=0)

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path, input_size):
        self.model = TemporalFusionTransformer(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers).to(self.device)
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        print(f"Model loaded from {path}")