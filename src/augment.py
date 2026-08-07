import torch

class TimeSeriesAugmenter:
    def __init__(self, device):
        self.device = device
        
    def jitter(self, x, sigma=0.01):
        # Noise is proportional to the standard deviation of each feature
        # This prevents breaking small-scale features like MACD
        feat_std = torch.std(x, dim=1, keepdim=True)
        feat_std = torch.where(feat_std > 1e-6, feat_std, torch.ones_like(feat_std))
        noise = torch.randn_like(x) * feat_std * sigma
        return x + noise
        
    def scaling(self, x, sigma=0.01):
        factor = torch.randn((x.shape[0], 1, x.shape[2])).to(self.device) * sigma + 1.0
        return x * factor
        
    def skew(self, x, max_skew=0.01):
        seq_len = x.shape[1]
        skew_factors = (torch.rand((x.shape[0], 1, x.shape[2])).to(self.device) * 2 - 1) * max_skew
        ramp = torch.linspace(-0.5, 0.5, seq_len).unsqueeze(0).unsqueeze(-1).to(self.device)
        return x + (x * ramp * skew_factors)

    def time_step_masking(self, x, mask_prob=0.05):
        """Randomly zeroes out entire time steps (candles) across the sequence."""
        # Create a mask of shape (batch, seq_len, 1)
        mask = (torch.rand((x.shape[0], x.shape[1], 1)).to(self.device) > mask_prob).float()
        return x * mask

    def feature_masking(self, x, mask_prob=0.05):
        """Randomly zeroes out specific features for the entire sequence."""
        # Create a mask of shape (batch, 1, num_features)
        mask = (torch.rand((x.shape[0], 1, x.shape[2])).to(self.device) > mask_prob).float()
        return x * mask

    def augment(self, x):
        """Apply a random combination of augmentations to generate a view."""
        x_aug = x.clone()
        
        # 1. Add Proportional Noise (Jitter, Scaling, Skew) - lower probability and intensity
        if torch.rand(1).item() > 0.6:
            x_aug = self.jitter(x_aug)
        if torch.rand(1).item() > 0.6:
            x_aug = self.scaling(x_aug)
        if torch.rand(1).item() > 0.6:
            x_aug = self.skew(x_aug)
            
        # 2. Add Masking (Dropout for robustness)
        if torch.rand(1).item() > 0.5:
            x_aug = self.time_step_masking(x_aug)
        if torch.rand(1).item() > 0.5:
            x_aug = self.feature_masking(x_aug)
        
        # Ensure at least one augmentation is applied if somehow all were skipped
        if torch.all(x_aug == x):
            x_aug = self.time_step_masking(x_aug)
            
        return x_aug
