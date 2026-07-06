import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import joblib
import os
from pathlib import Path

class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(TemporalFusionTransformer, self).__init__()
        
        # LSTM Encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Temporal Attention Layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # 3-Class Classifier (0: Flat, 1: Up, 2: Down)
        # Note: No Sigmoid/Softmax here; nn.CrossEntropyLoss expects raw logits
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3) 
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        final_timestep = attn_out[:, -1, :]
        out = self.fc_out(final_timestep)
        return out

class PricePredictor:
    def __init__(self, input_chunk_length=30, hidden_size=64, num_layers=2):
        self.input_chunk_length = input_chunk_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if not torch.cuda.is_available():
            raise RuntimeError("GPU requested but CUDA is not available!")
        self.device = torch.device('cuda')
        self.scaler = StandardScaler()
        self.model = None
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        # For Confusion Matrix
        self.last_val_targets = []
        self.last_val_preds = []

    def create_sequences(self, features_scaled, target):
        X, y = [], []
        for i in range(len(features_scaled) - self.input_chunk_length):
            X.append(features_scaled[i:i + self.input_chunk_length])
            y.append(target[i + self.input_chunk_length])
        return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(y))

    def train(self, train_csv, epochs=150, batch_size=64):
        df = pd.read_csv(train_csv, index_col=0, parse_dates=True)
        target = df['target'].values
        features = df.drop(columns=['target']).values
        
        # 80/20 Internal Train/Val split (done before scaling to prevent leakage)
        split = int(0.8 * len(features))
        
        features_train = features[:split]
        target_train = target[:split]
        
        # Include previous chunk_length rows so the first val prediction is exactly at `split`
        features_val = features[split - self.input_chunk_length:]
        target_val = target[split - self.input_chunk_length:]
        
        features_train_scaled = self.scaler.fit_transform(features_train)
        features_val_scaled = self.scaler.transform(features_val)
        
        X_train, y_train = self.create_sequences(features_train_scaled, target_train)
        X_val, y_val = self.create_sequences(features_val_scaled, target_val)
        
        X_train, y_train = X_train.to(self.device), y_train.to(self.device)
        X_val, y_val = X_val.to(self.device), y_val.to(self.device)
        
        input_features = features.shape[1]

        self.model = TemporalFusionTransformer(
            input_size=input_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        ).to(self.device)

        # CrossEntropyLoss automatically handles multi-class outputs
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = 15

        print(f"Training Multi-Class TFT (Features: {input_features}, Target Classes: 3)...")
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            # Shuffle Training Data
            indices = torch.randperm(X_train.size(0))
            X_train = X_train[indices]
            y_train = y_train[indices]

            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= (len(X_train) / batch_size)

            # Validation
            self.model.eval()
            val_loss = 0
            correct = 0
            all_preds, all_targets = [], []
            
            with torch.no_grad():
                for i in range(0, len(X_val), batch_size):
                    batch_X = X_val[i:i + batch_size]
                    batch_y = y_val[i:i + batch_size]
                    
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    # Calculate Multiclass Accuracy
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == batch_y).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(batch_y.cpu().numpy())

            val_loss /= (len(X_val) / batch_size)
            val_acc = correct / len(X_val)
            scheduler.step(val_loss)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # Save state for confusion matrix
            self.last_val_targets = all_targets
            self.last_val_preds = all_preds

            if epoch % 5 == 0 or epoch == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch+1:03d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {lr:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}. Restoring best weights.")
                self.model.load_state_dict(best_weights)
                break

        return train_loss, val_acc

    def predict(self, sequence):
        self.model.eval()
        with torch.no_grad():
            # FIX: Strip feature names if it's a Pandas DataFrame to suppress sklearn warnings
            if isinstance(sequence, (pd.DataFrame, pd.Series)):
                sequence = sequence.values
            elif not isinstance(sequence, np.ndarray):
                sequence = np.asarray(sequence)
                
            sequence_scaled = self.scaler.transform(sequence)
            seq_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0).to(self.device)
            outputs = self.model(seq_tensor)
            # Apply softmax to get pure probabilities for [P_Flat, P_Up, P_Down]
            probs = torch.softmax(outputs, dim=1)
            return probs.cpu().numpy()[0]

    def save(self, filepath):
        path_obj = Path(filepath)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), str(path_obj))
        scaler_path = path_obj.with_suffix('.scaler.pkl')
        joblib.dump(self.scaler, str(scaler_path))

    def plot_confusion_matrix(self, save_path=None):
        if not self.last_val_targets:
            print("No validation data available for Confusion Matrix.")
            return

        cm = confusion_matrix(self.last_val_targets, self.last_val_preds, labels=[0, 1, 2])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Flat (0)', 'Up (+1)', 'Down (-1)'],
                    yticklabels=['Flat (0)', 'Up (+1)', 'Down (-1)'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Triple Barrier Confusion Matrix')
        plt.tight_layout()
        
        if save_path:
            path_obj = Path(save_path).resolve()
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(path_obj), dpi=100)
            print(f"Confusion Matrix saved to {path_obj}")
        # plt.show()
        plt.close()

    def plot_training_history(self, save_path=None):
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Multi-Class Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Cross-Entropy Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            path_obj = Path(save_path).resolve()
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(path_obj), dpi=100)
        # plt.show()
        plt.close()