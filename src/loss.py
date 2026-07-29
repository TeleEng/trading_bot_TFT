import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveClusteringLoss(nn.Module):
    """
    Supervised Contrastive Clustering Loss (SupCon + Cluster).
    
    Instance-level: Pulls same-label samples together, pushes different-label
    samples apart. This makes embeddings label-aware from the start.
    
    Cluster-level: Same as before — ensures cluster assignments are consistent
    across augmentations and avoids collapse via entropy regularization.
    """
    def __init__(self, batch_size, instance_temp=0.1, cluster_temp=1.0):
        super(SupervisedContrastiveClusteringLoss, self).__init__()
        self.batch_size = batch_size
        self.instance_temp = instance_temp  # Lower temp sharpens SupCon
        self.cluster_temp = cluster_temp
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, z_i, z_j, c_i, c_j, labels):
        """
        Args:
            z_i, z_j: Instance embeddings from two augmented views (batch_size, dim)
            c_i, c_j: Cluster probabilities from two augmented views (batch_size, num_clusters)
            labels: Ground truth labels (batch_size,)
        """
        N = self.batch_size
        
        # ================= Supervised Contrastive (Instance) Loss =================
        # Stack both views: [z_i; z_j] -> (2N, dim)
        z = torch.cat((z_i, z_j), dim=0)
        z = F.normalize(z, dim=1)
        
        # Full pairwise similarity matrix (2N, 2N)
        sim = torch.matmul(z, z.T) / self.instance_temp
        
        # Build label mask: (2N, 2N) where mask[i,j]=1 if same label
        # Labels are duplicated for both views
        labels_2n = torch.cat((labels, labels), dim=0)  # (2N,)
        label_mask = (labels_2n.unsqueeze(0) == labels_2n.unsqueeze(1)).float()  # (2N, 2N)
        
        # Remove self-similarity from both the mask and the logits
        self_mask = torch.eye(2 * N, device=z.device)
        label_mask = label_mask * (1 - self_mask)  # exclude self
        
        # For numerical stability
        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()
        
        # Denominator: exp(sim) for all non-self pairs
        exp_logits = torch.exp(logits) * (1 - self_mask)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        
        # Mean of log-prob over positive pairs
        # Handle edge case where a sample has no positives (shouldn't happen with 3 classes)
        num_positives = label_mask.sum(dim=1)
        num_positives = torch.clamp(num_positives, min=1.0)
        
        mean_log_prob = (label_mask * log_prob).sum(dim=1) / num_positives
        loss_instance = -mean_log_prob.mean()

        # ================= Cluster-level Loss (unchanged) =================
        c_i_t = c_i.t()  # (num_clusters, batch_size)
        c_j_t = c_j.t()
        K = c_i_t.shape[0]
        
        c = torch.cat((c_i_t, c_j_t), dim=0)  # (2K, batch_size)
        c = F.normalize(c, dim=1)
        sim_cluster = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.cluster_temp
        
        sim_i_j_c = torch.diag(sim_cluster, K)
        sim_j_i_c = torch.diag(sim_cluster, -K)
        positives_c = torch.cat([sim_i_j_c, sim_j_i_c], dim=0).reshape(2*K, 1)
        
        mask_c = self._mask_correlated(K).to(c.device)
        negatives_c = sim_cluster[mask_c].reshape(2*K, -1)
        
        labels_c = torch.zeros(2*K, dtype=torch.long).to(c.device)
        logits_cluster = torch.cat((positives_c, negatives_c), dim=1)
        loss_cluster = F.cross_entropy(logits_cluster, labels_c, reduction='sum') / (2*K)
        
        # Entropy Regularization to prevent cluster collapse
        p = torch.cat((c_i, c_j), dim=0).mean(dim=0)
        entropy_loss = torch.sum(p * torch.log(p + 1e-8))

        return loss_instance + loss_cluster + entropy_loss

    def _mask_correlated(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=torch.bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask
