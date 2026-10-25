import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import os
from pathlib import Path
import gc

# Single source of truth for non-feature columns removed before sequence creation.
# Raw ATR / SMA_ATR_50 are non-stationary; sin/cos are collinear with Hour/DayOfWeek embeddings.
DROP_COLS = ['target_long', 'target_short', 'target', 'Open', 'High', 'Low', 'Close', 'Volume',
             'ATR', 'SMA_ATR_50', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']


class GLU(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, input_size * 2)
    def forward(self, x):
        return nn.functional.glu(self.fc(x), dim=-1)

class GRN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.glu = GLU(hidden_size)
        self.skip = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        skip = self.skip(x)
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.glu(x)
        return self.norm(skip + x)

class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, self.head_dim)
        self.out_proj = nn.Linear(self.head_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v):
        batch_size, seq_len, _ = q.size()
        Q = self.q_proj(q).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(k).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(v).unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V).mean(dim=1)
        return self.out_proj(context)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]

class MultiTimeframeTFT(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, cat_indices=None, dropout=0.35, num_classes=2):
        super(MultiTimeframeTFT, self).__init__()
        
        self.cat_indices = cat_indices if cat_indices is not None else []
        self.cont_indices = [i for i in range(input_size) if i not in self.cat_indices]
        cont_size = len(self.cont_indices)
        
        # Categorical Embeddings
        self.hour_embedding = nn.Embedding(24, hidden_size) if len(self.cat_indices) > 0 else None
        self.day_embedding = nn.Embedding(7, hidden_size) if len(self.cat_indices) > 1 else None
        
        # Continuous Feature Projection + LayerNorm across embedding dimension
        self.cont_proj = nn.Linear(cont_size, hidden_size)
        self.cont_norm = nn.LayerNorm(hidden_size)
        
        # GRN Encoders for each branch
        self.grn_1h = GRN(hidden_size, hidden_size, dropout=dropout)
        self.grn_4h = GRN(hidden_size, hidden_size, dropout=dropout)
        self.grn_1d = GRN(hidden_size, hidden_size, dropout=dropout)
        self.grn_1w = GRN(hidden_size, hidden_size, dropout=dropout)
        
        # LSTMs with residual skip connections
        self.lstm_1h = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.lstm_4h = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.lstm_1d = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.lstm_1w = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.branch_norm_1h = nn.LayerNorm(hidden_size)
        self.branch_norm_4h = nn.LayerNorm(hidden_size)
        self.branch_norm_1d = nn.LayerNorm(hidden_size)
        self.branch_norm_1w = nn.LayerNorm(hidden_size)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(hidden_size)
        
        # Interpretable Attention
        self.attention_1h = InterpretableMultiHeadAttention(embed_dim=hidden_size, num_heads=8, dropout=dropout)
        self.attn_layer_norm = nn.LayerNorm(hidden_size)
        
        # Cross-Timeframe Attention
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4, dropout=dropout, batch_first=True)
        self.cross_layer_norm = nn.LayerNorm(hidden_size)
        
        # Fusion Layer
        self.fusion = GRN(hidden_size * 4, hidden_size, dropout=dropout)
        
        # Supervised Classification Head
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def process_branch(self, x, grn_layer, lstm_layer, norm_layer):
        if len(self.cat_indices) > 0:
            cont_x = x[:, :, self.cont_indices]
            cat_x = x[:, :, self.cat_indices].long()
            
            x_emb = self.cont_norm(self.cont_proj(cont_x))
            x_emb += self.hour_embedding(cat_x[:, :, 0])
            if len(self.cat_indices) > 1:
                x_emb += self.day_embedding(cat_x[:, :, 1])
        else:
            x_emb = self.cont_norm(self.cont_proj(x))
            
        x_grn = grn_layer(x_emb)
        lstm_out, _ = lstm_layer(x_grn)
        return norm_layer(x_grn + lstm_out)

    def forward(self, x_1h, x_4h, x_1d, x_1w):
        out_1h = self.process_branch(x_1h, self.grn_1h, self.lstm_1h, self.branch_norm_1h)
        out_4h = self.process_branch(x_4h, self.grn_4h, self.lstm_4h, self.branch_norm_4h)
        out_1d = self.process_branch(x_1d, self.grn_1d, self.lstm_1d, self.branch_norm_1d)
        out_1w = self.process_branch(x_1w, self.grn_1w, self.lstm_1w, self.branch_norm_1w)
        
        # Positional Encoding + Attention
        out_1h_pos = self.pos_encoder(out_1h)
        attn_out = self.attention_1h(out_1h_pos, out_1h_pos, out_1h_pos)
        out_1h = self.attn_layer_norm(out_1h + attn_out)
        
        h_1h = out_1h[:, -1, :]
        h_4h = out_4h[:, -1, :]
        h_1d = out_1d[:, -1, :]
        h_1w = out_1w[:, -1, :]
        
        # Stack into [batch, 4_timeframes, hidden_size]
        stacked_h = torch.stack((h_1h, h_4h, h_1d, h_1w), dim=1)
        
        # Apply Cross-Timeframe Attention
        # Queries, Keys, and Values are all the stacked timeframes
        attn_out, _ = self.cross_attention(stacked_h, stacked_h, stacked_h)
        stacked_h = self.cross_layer_norm(stacked_h + attn_out)
        
        # Flatten back to [batch, hidden_size * 4] for Fusion GRN
        flattened_h = stacked_h.view(stacked_h.size(0), -1)
        
        h = self.fusion(flattened_h)
        logits = self.classification_head(h)
        return h, logits

class ProbabilityCalibrator:
    """
    Platt Scaling (logistic probability calibration: P(y=1|p) = sigmoid(a * logit(p) + b)).
    Monotonic, preserves ranking, and aligns model outputs with empirical validation base rates.

    Two output spaces are exposed:
      - predict(p)            -> TRUE calibrated probability P(TP hit) (use for EV-in-R maths).
      - predict_balanced(p)   -> PRIOR-CORRECTED probability where 0.50 == validation base rate.
                                 logit(p_bal) = logit(p_cal) - logit(base_rate)
                                 Long and short brains become directly comparable (no base-rate
                                 offset), so min_confidence=0.50 / min_edge gates measure LIFT.
    """
    def __init__(self):
        self.a = 1.0
        self.b = 0.0
        self.base_rate = 0.5

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        return np.log(p / (1.0 - p))

    def fit(self, probs, labels):
        logits = self._logit(probs)
        y = np.array(labels, dtype=np.float64)
        self.base_rate = float(np.clip(np.mean(y), 1e-4, 1.0 - 1e-4))
        
        try:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(C=1.0)
            clf.fit(logits.reshape(-1, 1), y)
            self.a = float(clf.coef_[0][0])
            self.b = float(clf.intercept_[0])
        except Exception:
            self.a = 1.0
            self.b = float(np.log(self.base_rate / (1.0 - self.base_rate)))

    def predict(self, probs):
        z = self.a * self._logit(probs) + self.b
        return 1.0 / (1.0 + np.exp(-z))

    def predict_balanced(self, probs):
        base_rate = getattr(self, 'base_rate', 0.5)
        z = self.a * self._logit(probs) + self.b - float(self._logit(base_rate))
        return 1.0 / (1.0 + np.exp(-z))


class PricePredictor:
    def __init__(self, input_chunk_length=52, hidden_size=128, num_layers=2, num_classes=2):
        self.input_chunk_length = input_chunk_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.calibrator = None
        self.model = None
        import copy
        self.copy = copy
        self.history = {'train_loss': [], 'val_loss': []}
        self.mean_1h = None; self.std_1h = None
        self.mean_4h = None; self.std_4h = None
        self.mean_1d = None; self.std_1d = None
        self.mean_1w = None; self.std_1w = None

    def create_sequences(self, df_1h, df_4h, df_1d, df_1w, fit_scaler=False, target_col='target_long'):
        drop_cols = DROP_COLS
        raw_1h = df_1h.drop(columns=drop_cols, errors='ignore')
        raw_4h = df_4h.drop(columns=drop_cols, errors='ignore')
        raw_1d = df_1d.drop(columns=drop_cols, errors='ignore')
        raw_1w = df_1w.drop(columns=drop_cols, errors='ignore')
        
        cat_cols = ['Hour', 'DayOfWeek']
        features_1h = raw_1h.values.astype(np.float32)
        features_4h = raw_4h.values.astype(np.float32)
        features_1d = raw_1d.values.astype(np.float32)
        features_1w = raw_1w.values.astype(np.float32)
        
        # Fit per-feature StandardScaler on training split to eliminate feature-magnitude disparity
        if fit_scaler:
            self.mean_1h = np.nanmean(features_1h, axis=0, keepdims=True)
            self.std_1h = np.nanstd(features_1h, axis=0, keepdims=True)
            self.std_1h = np.where(self.std_1h > 1e-4, self.std_1h, 1.0)
            
            self.mean_4h = np.nanmean(features_4h, axis=0, keepdims=True)
            self.std_4h = np.nanstd(features_4h, axis=0, keepdims=True)
            self.std_4h = np.where(self.std_4h > 1e-4, self.std_4h, 1.0)
            
            self.mean_1d = np.nanmean(features_1d, axis=0, keepdims=True)
            self.std_1d = np.nanstd(features_1d, axis=0, keepdims=True)
            self.std_1d = np.where(self.std_1d > 1e-4, self.std_1d, 1.0)
            
            self.mean_1w = np.nanmean(features_1w, axis=0, keepdims=True)
            self.std_1w = np.nanstd(features_1w, axis=0, keepdims=True)
            self.std_1w = np.where(self.std_1w > 1e-4, self.std_1w, 1.0)
            
        if self.mean_1h is not None:
            cat_idx_1h = [raw_1h.columns.get_loc(c) for c in cat_cols if c in raw_1h.columns]
            cat_idx_4h = [raw_4h.columns.get_loc(c) for c in cat_cols if c in raw_4h.columns]
            cat_idx_1d = [raw_1d.columns.get_loc(c) for c in cat_cols if c in raw_1d.columns]
            cat_idx_1w = [raw_1w.columns.get_loc(c) for c in cat_cols if c in raw_1w.columns]
            
            f_1h = (features_1h - self.mean_1h) / self.std_1h
            f_4h = (features_4h - self.mean_4h) / self.std_4h
            f_1d = (features_1d - self.mean_1d) / self.std_1d
            f_1w = (features_1w - self.mean_1w) / self.std_1w
            
            # Preserve raw integer values for categorical embeddings
            if len(cat_idx_1h) > 0:
                f_1h[:, cat_idx_1h] = features_1h[:, cat_idx_1h]
            if len(cat_idx_4h) > 0:
                f_4h[:, cat_idx_4h] = features_4h[:, cat_idx_4h]
            if len(cat_idx_1d) > 0:
                f_1d[:, cat_idx_1d] = features_1d[:, cat_idx_1d]
            if len(cat_idx_1w) > 0:
                f_1w[:, cat_idx_1w] = features_1w[:, cat_idx_1w]
        else:
            f_1h, f_4h, f_1d, f_1w = features_1h, features_4h, features_1d, features_1w
            
        target = df_1h[target_col].values if target_col in df_1h.columns else None
        
        X_1h, X_4h, X_1d, X_1w, y = [], [], [], [], []
        
        close_1h = df_1h.index + pd.Timedelta(hours=1)
        close_4h = df_4h.index + pd.Timedelta(hours=4)
        close_1d = df_1d.index + pd.Timedelta(days=1)
        close_1w = df_1w.index + pd.Timedelta(days=1)
        
        # Exact close-time matching: latest completed higher-timeframe bar whose close_tf <= close_1h
        idx_4h = np.searchsorted(close_4h.values, close_1h.values, side='right') - 1
        idx_1d = np.searchsorted(close_1d.values, close_1h.values, side='right') - 1
        idx_1w = np.searchsorted(close_1w.values, close_1h.values, side='right') - 1
        
        # Proper window slicing: includes the decision candle i at the end of the sequence for 1h,
        # and strictly completed bars up to decision time for higher timeframes (4h, 1d, 1w).
        for i in range(self.input_chunk_length - 1, len(df_1h)):
            i_4h = idx_4h[i]
            i_1d = idx_1d[i]
            i_1w = idx_1w[i]
            
            if i_4h >= 51 and i_1d >= 51 and i_1w >= 51:
                X_1h.append(f_1h[i - self.input_chunk_length + 1 : i + 1])
                X_4h.append(f_4h[i_4h - 52 + 1 : i_4h + 1])
                X_1d.append(f_1d[i_1d - 52 + 1 : i_1d + 1])
                X_1w.append(f_1w[i_1w - 52 + 1 : i_1w + 1])
                if target is not None:
                    y.append(target[i])
                    
        X_1h = np.nan_to_num(np.array(X_1h, dtype=np.float32), nan=0.0)
        X_4h = np.nan_to_num(np.array(X_4h, dtype=np.float32), nan=0.0)
        X_1d = np.nan_to_num(np.array(X_1d, dtype=np.float32), nan=0.0)
        X_1w = np.nan_to_num(np.array(X_1w, dtype=np.float32), nan=0.0)
        
        if target is not None:
            return X_1h, X_4h, X_1d, X_1w, np.array(y, dtype=np.int64)
            
        return X_1h, X_4h, X_1d, X_1w

    def get_aligned_df(self, df_1h, df_4h, df_1d, df_1w):
        """Returns the subset of df_1h that exactly matches the generated sequences/embeddings."""
        close_1h = df_1h.index + pd.Timedelta(hours=1)
        close_4h = df_4h.index + pd.Timedelta(hours=4)
        close_1d = df_1d.index + pd.Timedelta(days=1)
        close_1w = df_1w.index + pd.Timedelta(days=1)
        
        idx_4h = np.searchsorted(close_4h.values, close_1h.values, side='right') - 1
        idx_1d = np.searchsorted(close_1d.values, close_1h.values, side='right') - 1
        idx_1w = np.searchsorted(close_1w.values, close_1h.values, side='right') - 1
        
        valid_mask = np.zeros(len(df_1h), dtype=bool)
        for i in range(self.input_chunk_length - 1, len(df_1h)):
            if idx_4h[i] >= 51 and idx_1d[i] >= 51 and idx_1w[i] >= 51:
                valid_mask[i] = True
        return df_1h.iloc[valid_mask]

    def train(self, dfs_train, dfs_val, target_col='target_long', epochs=50, batch_size=256, patience=20, active_augs=None):
        df_1h_t, df_4h_t, df_1d_t, df_1w_t = dfs_train
        df_1h_v, df_4h_v, df_1d_v, df_1w_v = dfs_val

        X1_t, X4_t, X1d_t, X1w_t, y_t = self.create_sequences(df_1h_t, df_4h_t, df_1d_t, df_1w_t, fit_scaler=True, target_col=target_col)
        X1_v, X4_v, X1d_v, X1w_v, y_v = self.create_sequences(df_1h_v, df_4h_v, df_1d_v, df_1w_v, fit_scaler=False, target_col=target_col)

        X1_t = torch.FloatTensor(X1_t).to(self.device)
        X4_t = torch.FloatTensor(X4_t).to(self.device)
        X1d_t = torch.FloatTensor(X1d_t).to(self.device)
        X1w_t = torch.FloatTensor(X1w_t).to(self.device)
        y_t = torch.LongTensor(y_t).to(self.device)
        
        X1_v = torch.FloatTensor(X1_v).to(self.device)
        X4_v = torch.FloatTensor(X4_v).to(self.device)
        X1d_v = torch.FloatTensor(X1d_v).to(self.device)
        X1w_v = torch.FloatTensor(X1w_v).to(self.device)
        y_v = torch.LongTensor(y_v).to(self.device)

        input_size = X1_t.shape[2]
        
        # Identify Categorical Indices
        cat_cols = ['Hour', 'DayOfWeek']
        drop_cols = DROP_COLS
        clean_cols = df_1h_t.drop(columns=drop_cols, errors='ignore').columns
        cat_indices = [clean_cols.get_loc(c) for c in cat_cols if c in clean_cols]

        self.model = MultiTimeframeTFT(
            input_size=input_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers,
            cat_indices=cat_indices,
            num_classes=self.num_classes,
            dropout=0.35
        ).to(self.device)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for PyTorch training via DataParallel!")
            self.model = nn.DataParallel(self.model)

        optimizer = optim.Adam(self.model.parameters(), lr=3e-4, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        
        train_dataset = torch.utils.data.TensorDataset(X1_t, X4_t, X1d_t, X1w_t, y_t)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        val_dataset = torch.utils.data.TensorDataset(X1_v, X4_v, X1d_v, X1w_v, y_v)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        # Class-weighted cross-entropy to combat imbalance
        label_counts = torch.bincount(y_t, minlength=self.num_classes).float()
        class_weights = (1.0 / (label_counts + 1e-6))
        class_weights = class_weights / class_weights.sum() * float(self.num_classes)
        class_weights = class_weights.to(self.device)
        
        ce_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
        
        print(f"Class weights: {class_weights.cpu().numpy()}")
        print("Starting Clean Supervised MTF-TFT Classification Training...")
        
        best_val_f1 = -float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for b_x1, b_x4, b_x1d, b_x1w, b_y in train_loader:
                optimizer.zero_grad()
                
                # Direct clean supervised forward pass
                _, logits = self.model(b_x1, b_x4, b_x1d, b_x1w)
                loss = ce_criterion(logits, b_y)
                
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                optimizer.step()
                total_loss += loss.item()
                
            scheduler.step()
                
            avg_train_loss = total_loss / len(train_loader)
            self.history['train_loss'].append(avg_train_loss)
            
            self.model.eval()
            total_val_loss = 0
            correct = 0
            total_samples = 0
            all_preds = []
            all_labels = []
            val_emb_stds = []
            with torch.no_grad():
                for b_x1, b_x4, b_x1d, b_x1w, b_y in val_loader:
                    h_val, logits_val = self.model(b_x1, b_x4, b_x1d, b_x1w)
                    loss_ce = ce_criterion(logits_val, b_y)
                    total_val_loss += loss_ce.item()
                    
                    preds = logits_val.argmax(dim=1)
                    correct += (preds == b_y).sum().item()
                    total_samples += b_y.size(0)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(b_y.cpu().numpy())
                    val_emb_stds.append(h_val.std(dim=0).mean().item())
                    
            avg_val_loss = total_val_loss / len(val_loader)
            val_acc = correct / total_samples
            val_macro_f1 = f1_score(all_labels, all_preds, average='macro')
            val_prec_1 = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
            val_rec_1 = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
            pred_pos_pct = (np.array(all_preds) == 1).mean() * 100.0
            true_pos_pct = (np.array(all_labels) == 1).mean() * 100.0
            avg_emb_std = float(np.mean(val_emb_stds)) if len(val_emb_stds) > 0 else 0.0
            self.history['val_loss'].append(avg_val_loss)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | F1: {val_macro_f1:.4f} | Acc: {val_acc*100:4.1f}% | PredPos%: {pred_pos_pct:4.1f}% (True: {true_pos_pct:4.1f}%) | Prec: {val_prec_1*100:4.1f}% | Rec: {val_rec_1*100:4.1f}% | EmbStd: {avg_emb_std:.3f}")
                
            # Early stopping check based on Macro F1 (higher is better)
            import copy
            if val_macro_f1 > best_val_f1:
                best_val_f1 = val_macro_f1
                patience_counter = 0
                best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}!")
                if best_model_state is not None:
                    self.model.load_state_dict(best_model_state)
                    print("Restored best model from early stopping checkpoint (highest Val F1).")
                break

        if best_model_state is not None and patience_counter < patience:
            self.model.load_state_dict(best_model_state)
            print("Restored best model from early stopping checkpoint (highest Val F1).")
            
        print(f"Final Model - Train Loss: {avg_train_loss:.4f} | Val F1: {best_val_f1:.4f}")
        return avg_train_loss, best_val_f1

    def predict_batch_embeddings(self, dfs, batch_size=512):
        """Returns (N, hidden_size) fused embeddings."""
        self.model.eval()
        df_1h, df_4h, df_1d, df_1w = dfs
        result = self.create_sequences(df_1h, df_4h, df_1d, df_1w)
        X1, X4, X1d, X1w = result[0], result[1], result[2], result[3]
        
        if len(X1) == 0:
            return np.array([])
            
        X1 = torch.FloatTensor(X1).to(self.device)
        X4 = torch.FloatTensor(X4).to(self.device)
        X1d = torch.FloatTensor(X1d).to(self.device)
        X1w = torch.FloatTensor(X1w).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X1, X4, X1d, X1w)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_h = []
        with torch.no_grad():
            for b_x1, b_x4, b_x1d, b_x1w in loader:
                h, _ = self.model(b_x1, b_x4, b_x1d, b_x1w)
                all_h.append(h.cpu().numpy())
                
        return np.concatenate(all_h, axis=0)
        
    def predict_batch_classified(self, dfs, batch_size=512):
        """Returns (N, num_classes) softmax probabilities from the classification head."""
        self.model.eval()
        df_1h, df_4h, df_1d, df_1w = dfs
        result = self.create_sequences(df_1h, df_4h, df_1d, df_1w)
        X1, X4, X1d, X1w = result[0], result[1], result[2], result[3]
        
        if len(X1) == 0:
            return np.array([])
            
        X1 = torch.FloatTensor(X1).to(self.device)
        X4 = torch.FloatTensor(X4).to(self.device)
        X1d = torch.FloatTensor(X1d).to(self.device)
        X1w = torch.FloatTensor(X1w).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X1, X4, X1d, X1w)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_probs = []
        with torch.no_grad():
            for b_x1, b_x4, b_x1d, b_x1w in loader:
                _, logits = self.model(b_x1, b_x4, b_x1d, b_x1w)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
                
        return np.concatenate(all_probs, axis=0)

    def fit_calibrator(self, dfs_val, target_col='target_long'):
        """Fits Platt scaling calibrator on the validation split."""
        raw_probs = self.predict_batch_classified(dfs_val)[:, 1]
        aligned_df = self.get_aligned_df(*dfs_val)
        y_val = aligned_df[target_col].values
        self.calibrator = ProbabilityCalibrator()
        self.calibrator.fit(raw_probs, y_val)
        cal = self.calibrator.predict(raw_probs)
        bal = self.calibrator.predict_balanced(raw_probs)
        print(f"  [{target_col.upper()}] Platt Calibrator fitted: a={self.calibrator.a:.4f}, b={self.calibrator.b:.4f} (Base Rate: {self.calibrator.base_rate*100:.1f}%)")
        print(f"    Raw mean: {np.mean(raw_probs):.4f} | True-calibrated mean: {np.mean(cal):.4f} | Balanced (0.5 = base rate) mean: {np.mean(bal):.4f} | Balanced >0.5: {(bal > 0.5).mean()*100:.1f}% of bars")

    def predict_batch_calibrated(self, dfs, batch_size=512, balanced=False):
        """Returns (N, 2) calibrated probabilities [P(0), P(1)].

        balanced=False -> true calibrated P(TP hit), for EV-in-R calculations.
        balanced=True  -> prior-corrected probability (0.50 == validation base rate), for gating/obs.
        """
        raw_probs = self.predict_batch_classified(dfs, batch_size)[:, 1]
        if getattr(self, 'calibrator', None) is not None:
            if balanced:
                cal_p1 = self.calibrator.predict_balanced(raw_probs)
            else:
                cal_p1 = self.calibrator.predict(raw_probs)
            return np.column_stack([1.0 - cal_p1, cal_p1])
        return np.column_stack([1.0 - raw_probs, raw_probs])

    def save(self, path):
        base_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        torch.save({
            'state_dict': base_model.state_dict(),
            'cat_indices': base_model.cat_indices,
            'scalers': {
                'mean_1h': self.mean_1h, 'std_1h': self.std_1h,
                'mean_4h': self.mean_4h, 'std_4h': self.std_4h,
                'mean_1d': self.mean_1d, 'std_1d': self.std_1d,
                'mean_1w': self.mean_1w, 'std_1w': self.std_1w,
            },
            'calibrator': ({'a': self.calibrator.a, 'b': self.calibrator.b,
                            'base_rate': getattr(self.calibrator, 'base_rate', 0.5)}
                           if getattr(self, 'calibrator', None) is not None else None)
        }, path)

    def load(self, path, input_size):
        # weights_only=False: checkpoint stores NumPy scaler arrays (PyTorch >= 2.6 defaults to weights_only=True)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        cat_indices = checkpoint.get('cat_indices', [])
        scalers = checkpoint.get('scalers', {})
        self.mean_1h = scalers.get('mean_1h', None)
        self.std_1h = scalers.get('std_1h', None)
        self.mean_4h = scalers.get('mean_4h', None)
        self.std_4h = scalers.get('std_4h', None)
        self.mean_1d = scalers.get('mean_1d', None)
        self.std_1d = scalers.get('std_1d', None)
        self.mean_1w = scalers.get('mean_1w', None)
        self.std_1w = scalers.get('std_1w', None)
        cal_state = checkpoint.get('calibrator', None)
        if isinstance(cal_state, dict):
            self.calibrator = ProbabilityCalibrator()
            self.calibrator.a = float(cal_state.get('a', 1.0))
            self.calibrator.b = float(cal_state.get('b', 0.0))
            self.calibrator.base_rate = float(cal_state.get('base_rate', 0.5))
        else:
            self.calibrator = cal_state
        
        self.model = MultiTimeframeTFT(
            input_size=input_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers,
            cat_indices=cat_indices,
            num_classes=self.num_classes,
            dropout=0.35
        ).to(self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            
        self.model.eval()
        print(f"Model loaded from {path}")

    @staticmethod
    def save_cached_embeddings(path, embeddings):
        from pathlib import Path
        import numpy as np
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, embeddings)
        print(f"[CACHE] Saved embeddings to {p} (shape: {embeddings.shape})")

    @staticmethod
    def load_cached_embeddings(path):
        from pathlib import Path
        import numpy as np
        p = Path(path)
        emb = np.load(p)
        print(f"[CACHE] Loaded embeddings from {p} (shape: {emb.shape})")
        return emb
