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
from loss import SupervisedContrastiveClusteringLoss
from imblearn.under_sampling import EditedNearestNeighbours

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

class MultiTimeframeTFT(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_over_clusters=30, dropout=0.2):
        super(MultiTimeframeTFT, self).__init__()
        self.norm = InstanceNormalization1D()
        
        # 3 Variable Selection Networks
        self.vsn_1h = VariableSelectionNetwork(input_size, hidden_size, dropout)
        self.vsn_4h = VariableSelectionNetwork(input_size, hidden_size, dropout)
        self.vsn_1d = VariableSelectionNetwork(input_size, hidden_size, dropout)
        
        # 3 LSTMs
        self.lstm_1h = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.lstm_4h = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.lstm_1d = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # We will apply attention independently to the 1h sequence, as it's the most granular
        self.attention_1h = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4, dropout=dropout, batch_first=True)
        self.attn_layer_norm = nn.LayerNorm(hidden_size)
        
        # Fusion Layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Instance Head for InfoNCE
        self.instance_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 32)
        )
        
        # Over-Cluster Head (30 micro-clusters)
        self.num_over_clusters = num_over_clusters
        self.over_cluster_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, num_over_clusters),
            nn.Softmax(dim=-1)
        )
        
        # Direct Classification Head (3 classes: Flat/Up/Down)
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 3)
        )

    def process_branch(self, x, vsn_layer, lstm_layer):
        x = self.norm(x)
        x = vsn_layer(x)
        lstm_out, _ = lstm_layer(x)
        return lstm_out

    def forward(self, x_1h, x_4h, x_1d):
        # 1. Process branches
        out_1h = self.process_branch(x_1h, self.vsn_1h, self.lstm_1h)
        out_4h = self.process_branch(x_4h, self.vsn_4h, self.lstm_4h)
        out_1d = self.process_branch(x_1d, self.vsn_1d, self.lstm_1d)
        
        # 2. Attention on the 1h (base timeframe)
        attn_out, _ = self.attention_1h(out_1h, out_1h, out_1h)
        out_1h = self.attn_layer_norm(out_1h + attn_out)
        
        # 3. Extract final embeddings
        h_1h = out_1h[:, -1, :]
        h_4h = out_4h[:, -1, :]
        h_1d = out_1d[:, -1, :]
        
        # 4. Fuse
        h = self.fusion(torch.cat((h_1h, h_4h, h_1d), dim=1))
        
        # 5. Heads
        z = self.instance_head(h)
        c_over = self.over_cluster_head(h)
        logits = self.classification_head(h)
        
        return h, z, c_over, logits

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

    def create_sequences(self, df_1h, df_4h, df_1d, clean_noise=False):
        features_1h = df_1h.drop(columns=['target'], errors='ignore').values
        features_4h = df_4h.drop(columns=['target'], errors='ignore').values
        features_1d = df_1d.drop(columns=['target'], errors='ignore').values
        
        target = df_1h['target'].values if 'target' in df_1h.columns else None
        
        X_1h, X_4h, X_1d, y = [], [], [], []
        
        idx_4h = df_4h.index.get_indexer(df_1h.index, method='ffill')
        idx_1d = df_1d.index.get_indexer(df_1h.index, method='ffill')
        
        for i in range(self.input_chunk_length, len(df_1h)):
            i_4h = idx_4h[i]
            i_1d = idx_1d[i]
            
            if i_4h >= 15 and i_1d >= 5:
                X_1h.append(features_1h[i - self.input_chunk_length : i])
                X_4h.append(features_4h[i_4h - 15 : i_4h])
                X_1d.append(features_1d[i_1d - 5 : i_1d])
                if target is not None:
                    y.append(target[i])
                    
        X_1h, X_4h, X_1d = np.array(X_1h), np.array(X_4h), np.array(X_1d)
        
        if target is not None:
            y = np.array(y)
            if clean_noise:
                print("Applying EditedNearestNeighbours to clean noise...")
                X_enn = X_1h.reshape(X_1h.shape[0], -1)
                X_enn = np.nan_to_num(X_enn, nan=0.0)
                enn = EditedNearestNeighbours(n_neighbors=5, kind_sel='all')
                enn.fit_resample(X_enn, y)
                keep_idx = enn.sample_indices_
                
                noisy_samples = len(y) - len(keep_idx)
                print(f"Relabeled {noisy_samples} noisy/contradictory samples as 'Flat' (0) out of {len(y)}")
                
                noisy_mask = np.ones(len(y), dtype=bool)
                noisy_mask[keep_idx] = False
                y[noisy_mask] = 0
                
            return X_1h, X_4h, X_1d, y
            
        return X_1h, X_4h, X_1d

    def train(self, dfs_train, dfs_val, epochs=50, batch_size=256):
        df_1h_t, df_4h_t, df_1d_t = dfs_train
        df_1h_v, df_4h_v, df_1d_v = dfs_val

        X1_t, X4_t, X1d_t, y_t = self.create_sequences(df_1h_t, df_4h_t, df_1d_t, clean_noise=True)
        X1_v, X4_v, X1d_v, y_v = self.create_sequences(df_1h_v, df_4h_v, df_1d_v, clean_noise=False)

        X1_t = torch.FloatTensor(X1_t).to(self.device)
        X4_t = torch.FloatTensor(X4_t).to(self.device)
        X1d_t = torch.FloatTensor(X1d_t).to(self.device)
        y_t = torch.LongTensor(y_t).to(self.device)
        
        X1_v = torch.FloatTensor(X1_v).to(self.device)
        X4_v = torch.FloatTensor(X4_v).to(self.device)
        X1d_v = torch.FloatTensor(X1d_v).to(self.device)
        y_v = torch.LongTensor(y_v).to(self.device)

        input_size = X1_t.shape[2]
        self.model = MultiTimeframeTFT(
            input_size=input_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers
        ).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        
        train_dataset = torch.utils.data.TensorDataset(X1_t, X4_t, X1d_t, y_t)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        val_dataset = torch.utils.data.TensorDataset(X1_v, X4_v, X1d_v, y_v)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        criterion = SupervisedContrastiveClusteringLoss(batch_size=batch_size)
        
        # Class-weighted cross-entropy to combat the 53% Down imbalance
        # Compute inverse-frequency weights from training labels
        label_counts = torch.bincount(y_t, minlength=3).float()
        class_weights = (1.0 / (label_counts + 1e-6))
        class_weights = class_weights / class_weights.sum() * 3.0  # normalize so they sum to num_classes
        class_weights = class_weights.to(self.device)
        ce_criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        print(f"Class weights: {class_weights.cpu().numpy()}")
        print("Starting MTF Supervised Contrastive + Classification Training...")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            total_ce_loss = 0
            for b_x1, b_x4, b_x1d, b_y in train_loader:
                optimizer.zero_grad()
                
                # Augmentations on 1h (we can leave 4h and 1d unaugmented for stability)
                x1_i, x4_i, x1d_i = self.augmenter.augment(b_x1), b_x4, b_x1d
                x1_j, x4_j, x1d_j = self.augmenter.augment(b_x1), b_x4, b_x1d
                
                _, z_i, c_i, logits_i = self.model(x1_i, x4_i, x1d_i)
                _, z_j, c_j, logits_j = self.model(x1_j, x4_j, x1d_j)
                
                # SupCon + Cluster loss
                loss_contrastive = criterion(z_i, z_j, c_i, c_j, b_y)
                
                # Classification loss (average over both views)
                loss_ce = (ce_criterion(logits_i, b_y) + ce_criterion(logits_j, b_y)) / 2.0
                
                loss = loss_contrastive + loss_ce
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_ce_loss += loss_ce.item()
                
            avg_train_loss = total_loss / len(train_loader)
            avg_ce_loss = total_ce_loss / len(train_loader)
            self.history['train_loss'].append(avg_train_loss)
            
            self.model.eval()
            total_val_loss = 0
            total_val_ce = 0
            correct = 0
            total_samples = 0
            with torch.no_grad():
                for b_x1, b_x4, b_x1d, b_y in val_loader:
                    x1_i, x4_i, x1d_i = self.augmenter.augment(b_x1), b_x4, b_x1d
                    x1_j, x4_j, x1d_j = self.augmenter.augment(b_x1), b_x4, b_x1d
                    _, z_i, c_i, logits_i = self.model(x1_i, x4_i, x1d_i)
                    _, z_j, c_j, logits_j = self.model(x1_j, x4_j, x1d_j)
                    loss_contrastive = criterion(z_i, z_j, c_i, c_j, b_y)
                    loss_ce = (ce_criterion(logits_i, b_y) + ce_criterion(logits_j, b_y)) / 2.0
                    total_val_loss += (loss_contrastive + loss_ce).item()
                    total_val_ce += loss_ce.item()
                    # Accuracy on view i
                    preds = logits_i.argmax(dim=1)
                    correct += (preds == b_y).sum().item()
                    total_samples += b_y.size(0)
            
            avg_val_loss = total_val_loss / len(val_loader)
            self.history['val_loss'].append(avg_val_loss)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                val_acc = correct / max(total_samples, 1) * 100
                avg_val_ce = total_val_ce / len(val_loader)
                print(f"Epoch {epoch+1}/{epochs} | Train: {avg_train_loss:.4f} (CE: {avg_ce_loss:.4f}) | Val: {avg_val_loss:.4f} (CE: {avg_val_ce:.4f}) | Val Acc: {val_acc:.1f}%")

        return self.history['train_loss'][-1], self.history['val_loss'][-1]

    def set_voting_map(self, voting_map):
        self.voting_map = voting_map
        
    def _apply_voting(self, c_probs):
        # c_probs is (N, 30)
        # voting_map maps micro-cluster idx (0-29) to Triple Barrier Label (0=Flat, 1=Up, 2=Down)
        out_probs = np.zeros((len(c_probs), 3))
        for k in range(30):
            target_class = self.voting_map.get(k, 0)
            out_probs[:, target_class] += c_probs[:, k]
        return out_probs

    def predict(self, dfs):
        self.model.eval()
        df_1h, df_4h, df_1d = dfs
        result = self.create_sequences(df_1h, df_4h, df_1d)
        X1, X4, X1d = result[0], result[1], result[2]
        
        if len(X1) == 0:
            return 0.0, 0.0, 0.0
            
        X1 = torch.FloatTensor(X1[-1:]).to(self.device)
        X4 = torch.FloatTensor(X4[-1:]).to(self.device)
        X1d = torch.FloatTensor(X1d[-1:]).to(self.device)
        
        with torch.no_grad():
            _, _, _, logits = self.model(X1, X4, X1d)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            
        return probs[0, 0], probs[0, 1], probs[0, 2]

    def predict_batch(self, dfs, batch_size=512):
        """Returns (N, 30) raw micro-cluster probs for building the voting map."""
        self.model.eval()
        df_1h, df_4h, df_1d = dfs
        result = self.create_sequences(df_1h, df_4h, df_1d)
        X1, X4, X1d = result[0], result[1], result[2]
        
        if len(X1) == 0:
            return np.array([])
            
        X1 = torch.FloatTensor(X1).to(self.device)
        X4 = torch.FloatTensor(X4).to(self.device)
        X1d = torch.FloatTensor(X1d).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X1, X4, X1d)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_probs = []
        with torch.no_grad():
            for b_x1, b_x4, b_x1d in loader:
                _, _, c_over, _ = self.model(b_x1, b_x4, b_x1d)
                all_probs.append(c_over.cpu().numpy())
                
        c_probs = np.concatenate(all_probs, axis=0)
        return c_probs
        
    def predict_batch_classified(self, dfs, batch_size=512):
        """Returns (N, 3) softmax probabilities from the classification head."""
        self.model.eval()
        df_1h, df_4h, df_1d = dfs
        result = self.create_sequences(df_1h, df_4h, df_1d)
        X1, X4, X1d = result[0], result[1], result[2]
        
        if len(X1) == 0:
            return np.array([])
            
        X1 = torch.FloatTensor(X1).to(self.device)
        X4 = torch.FloatTensor(X4).to(self.device)
        X1d = torch.FloatTensor(X1d).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X1, X4, X1d)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_probs = []
        with torch.no_grad():
            for b_x1, b_x4, b_x1d in loader:
                _, _, _, logits = self.model(b_x1, b_x4, b_x1d)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
                
        return np.concatenate(all_probs, axis=0)

    def predict_batch_voted(self, dfs, batch_size=512):
        """Use classification head directly (bypasses cluster voting)."""
        return self.predict_batch_classified(dfs, batch_size)

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path, input_size):
        self.model = MultiTimeframeTFT(input_size=input_size, hidden_size=self.hidden_size, num_layers=self.num_layers).to(self.device)
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        print(f"Model loaded from {path}")