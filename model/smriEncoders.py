import torch
import torch.nn as nn
import torch_geometric as tg
from torch_geometric.nn import global_mean_pool



# simple mlp
class SMRIEncoder(nn.Module):
    
    def __init__(
        self,
        smri_dim,
        embed_dim=64
    ):
        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(smri_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, embed_dim)
        )

    def forward(self, x):

        return self.encoder(x)
# class SMRIEncoder(nn.Module):
#     def __init__(self, input_dim, dropout=0.3):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 256),
#             nn.ReLU(),
#             nn.Dropout(0.3),

#             nn.Linear(256, 64),
#             nn.ReLU()
#         )
        
#     def forward(self, x):
#         return self.net(x)  
    
    
# Attention    
class SMRIAttentionEncoder(nn.Module):
    
    def __init__(self, input_dim):
        super().__init__()

        self.feature_embed = nn.Linear(1, 32)

        self.attention = nn.MultiheadAttention(
            embed_dim=32,
            num_heads=4,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):


        x = x.unsqueeze(-1)     

        x = self.feature_embed(x)  

        attn_out, _ = self.attention(
            x, x, x
        )

        pooled = attn_out.mean(dim=1)

        return self.fc(pooled)
    
    
    
    

# multiview gcn

class LearnableEdgeWeight(nn.Module):
    def __init__(self, in_c, hidden_c):
        super().__init__()
        self.proj = nn.Linear(in_c, hidden_c, bias=False)
        self.q = nn.Linear(hidden_c, hidden_c, bias=False)
        self.k = nn.Linear(hidden_c, hidden_c, bias=False)
        self.scale = hidden_c ** -0.5

    def forward(self, x, edge_index):
        z = torch.tanh(self.proj(x))
        q = self.q(z)
        k = self.k(z)
        src, dst = edge_index[0], edge_index[1]
        score = (q[src] * k[dst]).sum(dim=-1) * self.scale
        reverse_score = (q[dst] * k[src]).sum(dim=-1) * self.scale
        score = 0.5 * (score + reverse_score)
        edge_weight = torch.sigmoid(score)
        edge_weight = torch.where(src == dst, torch.ones_like(edge_weight), edge_weight)
        return edge_weight


def make_complete_edge_index(num_nodes):
    src, dst = torch.meshgrid(
        torch.arange(num_nodes), torch.arange(num_nodes), indexing='ij')
    return torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0).long()


class ViewGCNBranch(nn.Module):
  
    def __init__(self, in_c, hid_c, K, dropout_rate, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        self.register_buffer('base_edge_index', make_complete_edge_index(num_nodes))

        self.edge_learner = LearnableEdgeWeight(in_c, hid_c)
        self.conv1 = tg.nn.ChebConv(in_c, hid_c, K, normalization='sym', bias=True)
        self.conv2 = tg.nn.ChebConv(hid_c, hid_c, K, normalization='sym', bias=True)
        self.conv3 = tg.nn.ChebConv(hid_c, hid_c, K, normalization='sym', bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: (B, N, F)
        B, N, F = x.shape
        device = x.device

        offsets = (torch.arange(B, device=device) * N).view(-1, 1, 1)
        edge_index = self.base_edge_index.to(device).unsqueeze(0).repeat(B, 1, 1) + offsets
        edge_index = edge_index.permute(1, 0, 2).reshape(2, -1)
        batch_vec = torch.arange(B, device=device).view(-1, 1).repeat(1, N).reshape(-1)

        x_flat = x.reshape(B * N, F)
        edge_weight = self.edge_learner(x_flat, edge_index)

        h = self.relu(self.dropout(self.conv1(x_flat, edge_index, edge_weight)))
        h = self.relu(self.dropout(self.conv2(h, edge_index, edge_weight)))
        h = self.relu(self.dropout(self.conv3(h, edge_index, edge_weight)))

        h = global_mean_pool(h, batch_vec)  # (B, hid_c)
        return h


class MultiViewSMRIEncoder(nn.Module):

    def __init__(self, in_c_by_view, num_nodes_by_view, hid_c=64, K=2, dropout_rate=0.3):
        super().__init__()
        self.view_names = ['aseg', 'destrieux', 'wmparc']
        self.branches = nn.ModuleDict({
            view: ViewGCNBranch(
                in_c=in_c_by_view[view],
                hid_c=hid_c,
                K=K,
                dropout_rate=dropout_rate,
                num_nodes=num_nodes_by_view[view]
            )
            for view in self.view_names
        })
        self.fusion = nn.Linear(hid_c * len(self.view_names), hid_c)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x_aseg, x_destrieux, x_wmparc):
        h_aseg = self.branches['aseg'](x_aseg)
        h_destrieux = self.branches['destrieux'](x_destrieux)
        h_wmparc = self.branches['wmparc'](x_wmparc)
        h = torch.cat([h_aseg, h_destrieux, h_wmparc], dim=1)
        h = self.relu(self.dropout(self.fusion(h)))
        return h  