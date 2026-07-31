import torch
import numpy as np

class PyTorchENN:
    """
    A GPU-accelerated implementation of Edited Nearest Neighbours (ENN).
    Uses batching to prevent VRAM overflow when calculating massive distance matrices.
    """
    def __init__(self, n_neighbors=5, batch_size=1024, device='cuda'):
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.device = device if torch.cuda.is_available() else 'cpu'

    def fit_resample(self, X, y):
        """
        X: numpy array (N, D)
        y: numpy array (N,)
        Returns: keep_idx (numpy array of indices to keep)
        """
        N = X.shape[0]
        # Move entire dataset to device (usually easily fits in VRAM for sizes < 100k)
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.long, device=self.device)
        
        keep_mask = torch.ones(N, dtype=torch.bool, device=self.device)
        
        print(f"Running PyTorch ENN on {self.device} with {N} samples in batches of {self.batch_size}...")
        
        for i in range(0, N, self.batch_size):
            end = min(i + self.batch_size, N)
            X_batch = X_tensor[i:end]
            
            # Compute Euclidean distance from current batch to all points in dataset
            # dist shape: (batch_size, N)
            dist = torch.cdist(X_batch, X_tensor)
            
            # Find the k+1 nearest neighbors (the closest one is the point itself)
            # We want smallest distances, so largest=False
            topk_dist, topk_idx = torch.topk(dist, k=self.n_neighbors + 1, dim=1, largest=False)
            
            # Exclude the first index (distance 0 to self)
            neighbor_idx = topk_idx[:, 1:] # shape: (batch_size, k)
            
            # Get the labels of those nearest neighbors
            neighbor_labels = y_tensor[neighbor_idx]
            batch_y = y_tensor[i:end].unsqueeze(1) # shape: (batch_size, 1)
            
            # kind_sel='all': a sample is kept ONLY if ALL its k neighbors match its label.
            # If any neighbor disagrees, it is considered noisy and removed.
            matches = (neighbor_labels == batch_y)
            all_match = matches.all(dim=1) # shape: (batch_size,)
            
            keep_mask[i:end] = all_match
            
        keep_idx = torch.where(keep_mask)[0].cpu().numpy()
        return keep_idx
