import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveClusteringLoss(nn.Module):
    def __init__(self, batch_size, instance_temp=0.5, cluster_temp=1.0):
        super(ContrastiveClusteringLoss, self).__init__()
        self.batch_size = batch_size
        self.instance_temp = instance_temp
        self.cluster_temp = cluster_temp
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.similarity_f = nn.CosineSimilarity(dim=2)
        
        self.mask = self.mask_correlated_samples(batch_size)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=torch.bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j, c_i, c_j):
        N = 2 * self.batch_size
        
        # ================= Instance-level Loss =================
        z = torch.cat((z_i, z_j), dim=0) # (2N, dim)
        z = F.normalize(z, dim=1)
        sim_instance = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.instance_temp
        
        sim_i_j = torch.diag(sim_instance, self.batch_size)
        sim_j_i = torch.diag(sim_instance, -self.batch_size)
        positives = torch.cat([sim_i_j, sim_j_i], dim=0).reshape(N, 1)
        
        mask = self.mask.to(z.device)
        negatives = sim_instance[mask].reshape(N, -1)
        
        labels = torch.zeros(N, dtype=torch.long).to(z.device)
        logits_instance = torch.cat((positives, negatives), dim=1)
        loss_instance = self.criterion(logits_instance, labels) / N

        # ================= Cluster-level Loss =================
        # Treat cluster columns as cluster representations
        c_i_t = c_i.t() # (num_clusters, batch_size)
        c_j_t = c_j.t()
        K = c_i_t.shape[0]
        
        c = torch.cat((c_i_t, c_j_t), dim=0) # (2K, batch_size)
        c = F.normalize(c, dim=1)
        sim_cluster = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.cluster_temp
        
        sim_i_j_c = torch.diag(sim_cluster, K)
        sim_j_i_c = torch.diag(sim_cluster, -K)
        positives_c = torch.cat([sim_i_j_c, sim_j_i_c], dim=0).reshape(2*K, 1)
        
        mask_c = self.mask_correlated_samples(K).to(c.device)
        negatives_c = sim_cluster[mask_c].reshape(2*K, -1)
        
        labels_c = torch.zeros(2*K, dtype=torch.long).to(c.device)
        logits_cluster = torch.cat((positives_c, negatives_c), dim=1)
        loss_cluster = self.criterion(logits_cluster, labels_c) / (2*K)
        
        # Entropy Regularization to prevent cluster collapse
        p = torch.cat((c_i, c_j), dim=0).mean(dim=0)
        entropy_loss = torch.sum(p * torch.log(p + 1e-8))

        return loss_instance + loss_cluster + entropy_loss
