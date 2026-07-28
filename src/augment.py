import torch

class TimeSeriesAugmenter:
    def __init__(self, device):
        self.device = device
        
    def jitter(self, x, sigma=0.03):
        noise = torch.randn_like(x) * sigma
        return x + noise
        
    def scaling(self, x, sigma=0.05):
        factor = torch.randn((x.shape[0], 1, x.shape[2])).to(self.device) * sigma + 1.0
        return x * factor
        
    def skew(self, x, max_skew=0.05):
        seq_len = x.shape[1]
        skew_factors = (torch.rand((x.shape[0], 1, x.shape[2])).to(self.device) * 2 - 1) * max_skew
        ramp = torch.linspace(-0.5, 0.5, seq_len).unsqueeze(0).unsqueeze(-1).to(self.device)
        return x + (x * ramp * skew_factors)

    def augment(self, x):
        """Apply a random combination of augmentations to generate a view."""
        x_aug = x.clone()
        if torch.rand(1).item() > 0.5:
            x_aug = self.jitter(x_aug)
        if torch.rand(1).item() > 0.5:
            x_aug = self.scaling(x_aug)
        if torch.rand(1).item() > 0.5:
            x_aug = self.skew(x_aug)
        
        # Ensure at least one augmentation is applied
        if torch.all(x_aug == x):
            x_aug = self.jitter(x_aug)
            
        return x_aug
