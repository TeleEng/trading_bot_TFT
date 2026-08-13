import torch
import torch.nn.functional as F

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
        mask = (torch.rand((x.shape[0], 1, x.shape[2]), device=self.device) > mask_prob).float()
        return x * mask

    def block_masking(self, x, mask_len=3):
        """Randomly zeroes out a contiguous block of time steps."""
        batch_size, seq_len, num_feats = x.shape
        mask = torch.ones((batch_size, seq_len, 1), device=self.device)
        # Random start indices for each sample in the batch
        starts = torch.randint(0, seq_len - mask_len + 1, (batch_size,), device=self.device)
        for i in range(batch_size):
            mask[i, starts[i]:starts[i]+mask_len, :] = 0.0
        return x * mask

    def magnitude_warping(self, x, sigma=0.1, knots=4):
        """Generates a random cubic spline hovering around 1.0 to multiply the sequence by."""
        batch_size, seq_len, num_feats = x.shape
        # Create random knots
        random_knots = torch.randn((batch_size, 1, knots), device=self.device) * sigma + 1.0
        # Interpolate to full sequence length (Linear interpolation handles batched 1D smoothly)
        curve = F.interpolate(random_knots, size=seq_len, mode='linear', align_corners=True)
        # Transpose back to (Batch, SeqLen, 1)
        curve = curve.transpose(1, 2)
        return x * curve

    def random_smoothing(self, x, kernel_size=3):
        """Applies 1D Average Pooling to smooth out high-frequency noise."""
        # avg_pool1d expects (Batch, Channels, Length), so we transpose (B, T, F) -> (B, F, T)
        x_t = x.transpose(1, 2)
        # Pad replication to maintain sequence length
        padding = kernel_size // 2
        x_padded = F.pad(x_t, (padding, padding), mode='replicate')
        x_smooth = F.avg_pool1d(x_padded, kernel_size=kernel_size, stride=1)
        return x_smooth.transpose(1, 2)

    def augment(self, x, active_augs=None):
        """
        Apply a random combination of augmentations to generate a view.
        If active_augs is provided (list of strings), only those augmentations are applied stochastically.
        """
        x_aug = x.clone()
        
        # Define available augmentations and their default probabilities
        # This controls the stochastic selection of views.
        all_augs = {
            'jitter': (self.jitter, 0.4),
            'scaling': (self.scaling, 0.3),
            'skew': (self.skew, 0.3),
            'time_step_masking': (self.time_step_masking, 0.3),
            'feature_masking': (self.feature_masking, 0.2),
            'block_masking': (self.block_masking, 0.4),
            'magnitude_warping': (self.magnitude_warping, 0.4),
            'random_smoothing': (self.random_smoothing, 0.3)
        }
        
        # If no specific augmentations requested, allow all to be possible
        if active_augs is None:
            active_augs = list(all_augs.keys())
            
        applied_any = False
        
        for aug_name in active_augs:
            if aug_name in all_augs:
                func, prob = all_augs[aug_name]
                # Stochastically apply this augmentation
                if torch.rand(1).item() < prob:
                    x_aug = func(x_aug)
                    applied_any = True
        
        # Ensure at least one augmentation is applied (Contrastive Learning requires distinct views)
        if not applied_any and len(active_augs) > 0:
            # Force apply the first active augmentation
            func, _ = all_augs[active_augs[0]]
            x_aug = func(x_aug)
            
        return x_aug
