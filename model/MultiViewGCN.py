import numpy as np
import torch
import torch.nn as nn
import torch_geometric as tg
from torch_geometric.utils import remove_self_loops, add_self_loops, scatter as pyg_scatter
try:
    from torch_geometric.utils import maybe_num_nodes
except ImportError:
    from torch_geometric.utils.num_nodes import maybe_num_nodes
try:
    from torch_geometric.nn.aggr import AttentionalAggregation
except ImportError:  # نسخه‌های قدیمی PyG
    from torch_geometric.nn import GlobalAttention as AttentionalAggregation

CONV_TYPES = ('cheb', 'gcn', 'graph', 'gat', 'gin', 'sage', 'tag', 'sgc', 'arma', 'bern')
GAT_HEADS = 1
FUSION_TYPES = ('concat', 'sum', 'mean', 'attention', 'gated')
LEARNABLE_GRAPH_MODES = ('learnable', 'learnable_scratch')
EDGE_SCRATCH_INIT_SCALE = 0.1
EDGE_DEGREE_EPS = 1e-6


import torch.nn.functional as F

def build_per_subject_graph(x_bnf, proj, k):
    B, n, _ = x_bnf.shape
    dev = x_bnf.device
    z = F.normalize(proj(x_bnf), dim=-1)                 # (B, n, d)
    sim = torch.bmm(z, z.transpose(1, 2))                # (B, n, n) cosine
    eye = torch.eye(n, dtype=torch.bool, device=dev).unsqueeze(0)
    sim = sim.masked_fill(eye, float('-inf'))            # بدون self-loop

    k = min(k, n - 1)
    vals, idx = sim.topk(k, dim=-1)                      # (B, n, k)

    offs = (torch.arange(B, device=dev) * n).view(B, 1, 1)
    src = torch.arange(n, device=dev).view(1, n, 1).expand(B, n, k) + offs
    dst = idx + offs

    edge_index = torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)
    edge_weight = ((vals + 1) / 2).reshape(-1) + 1e-6    # [0,1]، مثبت، gradient-friendly

    # symmetrize
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_weight = torch.cat([edge_weight, edge_weight], dim=0)
    return edge_index, edge_weight
























def build_view_conv(conv_type, in_c, hid_c, K):
    if conv_type == 'cheb':
        return SafeChebConv(in_c, hid_c, K, normalization='sym', bias=True)
    elif conv_type == 'gcn':
        return tg.nn.GCNConv(in_c, hid_c, add_self_loops=True, normalize=True, bias=True)
    elif conv_type == 'graph':
        return tg.nn.GraphConv(in_c, hid_c, aggr='add', bias=True)
    elif conv_type == 'gat':
        return tg.nn.GATConv(in_c, hid_c, heads=GAT_HEADS, concat=False,
                              edge_dim=1, dropout=0.0, bias=True)
    elif conv_type == 'gin':
        mlp = nn.Sequential(nn.Linear(in_c, hid_c), nn.ReLU(), nn.Linear(hid_c, hid_c))
        return tg.nn.GINEConv(mlp, edge_dim=1)
    elif conv_type == 'sage':
        return tg.nn.SAGEConv(in_c, hid_c, aggr='mean', bias=True)
    elif conv_type == 'sgc':
        return tg.nn.SGConv(in_c, hid_c, K=K, cached=False, bias=True)
    elif conv_type == 'tag':
        return tg.nn.TAGConv(in_c, hid_c, K=K, bias=True)
    elif conv_type == 'arma':
        return tg.nn.ARMAConv(in_c, hid_c, num_stacks=1, num_layers=K, shared_weights=True, dropout=0.0, bias=True)
    elif conv_type == 'bern':
        return tg.nn.BernConv(in_c, hid_c, K=K, bias=True)
    else:
        raise ValueError(f"conv_type must be one of {CONV_TYPES}, got: {conv_type!r}")


def run_view_conv(conv, conv_type, x, edge_index, edge_weight):
    if conv_type in ('cheb', 'gcn', 'graph'):
        return conv(x, edge_index, edge_weight)
    elif conv_type in ('gat', 'gin'):
        return conv(x, edge_index, edge_attr=edge_weight.unsqueeze(-1))
    elif conv_type == 'sage':
        return conv(x, edge_index)
    elif conv_type in ('sgc', 'arma', 'tag', 'bern'):
        return conv(x, edge_index, edge_weight)
    else:
        raise ValueError(f"conv_type must be one of {CONV_TYPES}, got: {conv_type!r}")


def safe_get_laplacian(edge_index, edge_weight=None, normalization=None,
                        dtype=None, num_nodes=None):
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=dtype, device=edge_index.device)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index[0], edge_index[1]
    deg = pyg_scatter(edge_weight, row, 0, dim_size=num_nodes, reduce='sum')
    deg_safe = deg.abs().clamp_min(EDGE_DEGREE_EPS)

    if normalization is None:
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        edge_weight = torch.cat([-edge_weight, deg_safe], dim=0)
    elif normalization == 'sym':
        deg_inv_sqrt = deg_safe.pow(-0.5)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        edge_index, edge_weight = add_self_loops(edge_index, -edge_weight, fill_value=1., num_nodes=num_nodes)
    else:  # 'rw'
        deg_inv = 1.0 / deg_safe
        edge_weight = deg_inv[row] * edge_weight
        edge_index, edge_weight = add_self_loops(edge_index, -edge_weight, fill_value=1., num_nodes=num_nodes)

    return edge_index, edge_weight


class SafeChebConv(tg.nn.ChebConv):
    """ChebConv whose normalization can never NaN, even with signed/learnable
    edge weights. Identical to tg.nn.ChebConv otherwise."""

    def __norm__(self, edge_index, num_nodes, edge_weight, normalization,
                 lambda_max=None, dtype=None, batch=None):
        edge_index, edge_weight = safe_get_laplacian(
            edge_index, edge_weight, normalization, dtype, num_nodes)
        assert edge_weight is not None

        if lambda_max is None:
            lambda_max = 2.0 * edge_weight.max()
        elif not isinstance(lambda_max, torch.Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=dtype, device=edge_index.device)
        assert lambda_max is not None

        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight.masked_fill_(edge_weight == float('inf'), 0)

        loop_mask = edge_index[0] == edge_index[1]
        edge_weight[loop_mask] -= 1

        return edge_index, edge_weight


def build_fusion_module(fusion_type, hid_c, n_views):
    if fusion_type in ('concat', 'sum', 'mean'):
        return None
    elif fusion_type == 'attention':
        return nn.Linear(hid_c, 1)
    elif fusion_type == 'gated':
        return nn.Linear(hid_c * n_views, hid_c * n_views)
    else:
        raise ValueError(f"fusion_type must be one of {FUSION_TYPES}, got: {fusion_type!r}")


def fuse_view_embeddings(fusion_type, fusion_module, view_embeddings):
    if fusion_type == 'concat':
        return torch.cat(view_embeddings, dim=1)

    stacked = torch.stack(view_embeddings, dim=1)  # (n_subjects, n_views, hid_c)

    if fusion_type == 'sum':
        return stacked.sum(dim=1)
    elif fusion_type == 'mean':
        return stacked.mean(dim=1)
    elif fusion_type == 'attention':
        scores = fusion_module(stacked).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (weights * stacked).sum(dim=1)
    elif fusion_type == 'gated':
        n_subjects, n_views, hid_c = stacked.shape
        concat_all = stacked.reshape(n_subjects, n_views * hid_c)
        gates = torch.sigmoid(fusion_module(concat_all))
        gated = (concat_all * gates).reshape(n_subjects, n_views, hid_c)
        return gated.sum(dim=1)
    else:
        raise ValueError(f"fusion_type must be one of {FUSION_TYPES}, got: {fusion_type!r}")


def tile_graph_for_batch(edge_index, edge_weight, n_nodes, n_subjects):
    """Repeats one shared per-view brain-region graph once per subject
    (block-diagonal), for the n_subjects actually present in THIS batch."""
    offsets = (np.arange(n_subjects) * n_nodes).reshape(-1, 1, 1)
    tiled = edge_index[None, :, :] + offsets
    edge_index_b = tiled.transpose(1, 0, 2).reshape(2, -1).astype(np.int64)
    edge_weight_b = np.tile(edge_weight, n_subjects).astype(np.float32)
    return edge_index_b, edge_weight_b


class MultiViewGCN(nn.Module):
    """Multi-view GCN sMRI encoder (aseg / aparc / wmparc).

    Unlike SMRIFCNEncoder / SMRITransformerEncoder this does NOT take a flat
    (B, D) tensor. Its input is per-ROI node features per view (see
    DHGFormer._forward_mvgcn for how the flat batch tensor coming out of the
    DataLoader is reshaped back into this dict). Graph topology/weights are
    computed once per fold from TRAIN-ONLY subjects (see
    imports/smri_graph_build.py) and passed in at construction time via
    base_edge_index / base_edge_weight; they are re-tiled dynamically inside
    forward_features() to match whatever batch size actually arrives, so this
    plugs into the existing mini-batch DataLoader / mixup training loop
    unchanged.

    forward_features() returns the FUSED EMBEDDING (no final classifier),
    exactly like SMRIFCNEncoder/SMRITransformerEncoder, so DHGFormer's
    existing fusion_classifier / ModalityAttentionFusion consume it unchanged.
    """

    def __init__(self, view_names, n_nodes_per_view, n_subfeat_per_view, hid_c,
                 K, dropout_rate, base_edge_index, base_edge_weight,
                 graph_mode='static', conv_type='cheb', fusion_type='concat',
                 k_per_view=None, graph_proj_dim=16, num_layers=1, pool_type='mean', extra_dim=0, extra_hid=16):
        super().__init__()
        self.view_names = view_names
        self.n_views = len(view_names)
        self.n_nodes_per_view = n_nodes_per_view
        self.graph_mode = graph_mode
        self.conv_type = conv_type
        if self.conv_type not in CONV_TYPES:
            raise ValueError(f"conv_type must be one of {CONV_TYPES}, got: {conv_type!r}")
        self.fusion_type = fusion_type
        if self.fusion_type not in FUSION_TYPES:
            raise ValueError(f"fusion_type must be one of {FUSION_TYPES}, got: {fusion_type!r}")


        self.num_layers = {
            v: (num_layers[v] if isinstance(num_layers, dict) else num_layers)
            for v in view_names
        }
        for v, n in self.num_layers.items():
            if n < 1:
                raise ValueError(f"num_layers for view {v!r} must be >= 1, got {n}")

        self.view_convs = nn.ModuleDict({
            view: nn.ModuleList([
                build_view_conv(conv_type,
                                n_subfeat_per_view[view] if i == 0 else hid_c,
                                hid_c, K)
                for i in range(self.num_layers[view])
            ])
            for view in view_names
        })
        
        self.pool_type = pool_type
        if pool_type == 'attention':
            self.att_pool = nn.ModuleDict({
                view: AttentionalAggregation(gate_nn=nn.Sequential(
                    nn.Linear(hid_c, hid_c // 2),
                    nn.ReLU(),
                    nn.Linear(hid_c // 2, 1)))
                for view in view_names
            })
        elif pool_type != 'mean':
            raise ValueError(f"pool_type must be 'mean' or 'attention', got: {pool_type!r}")
                
                
        # self.view_bns = nn.ModuleDict({
        #     view: nn.ModuleList([
        #         nn.BatchNorm1d(hid_c) for _ in range(self.num_layers[view])
        #     ])
        #     for view in view_names
        # })
                
        
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fusion_module = build_fusion_module(fusion_type, hid_c, self.n_views)
        # 'concat' -> hid_c * n_views ; every other fusion_type -> hid_c.
        # DHGFormer reads this to size fusion_classifier's input.
        self.out_dim = hid_c * self.n_views if fusion_type == 'concat' else hid_c
        self.extra_dim = extra_dim
        if extra_dim > 0:
            self.extra_mlp = nn.Sequential(
                nn.Linear(extra_dim, extra_hid), nn.ReLU(inplace=True), nn.Dropout(dropout_rate))
            self.out_dim += extra_hid
        # Untiled (single-copy) topology/weights per view -- plain numpy, not
        # buffers/parameters: re-tiled to the actual batch size on every
        # forward_features() call (see below), and moved to the right device
        # there via torch.as_tensor(..., device=x.device).
        self.base_edge_index = {v: np.asarray(base_edge_index[v], dtype=np.int64) for v in view_names}
        self.base_edge_weight = {v: np.asarray(base_edge_weight[v], dtype=np.float32) for v in view_names}

        if self.graph_mode in LEARNABLE_GRAPH_MODES:
            if self.graph_mode == 'learnable':
                self.edge_weight_params = nn.ParameterDict({
                    view: nn.Parameter(torch.as_tensor(base_edge_weight[view], dtype=torch.float32).clone())
                    for view in view_names
                })
            else:  # 'learnable_scratch'
                self.edge_weight_params = nn.ParameterDict({
                    view: nn.Parameter(
                        torch.empty(len(base_edge_weight[view]), dtype=torch.float32).uniform_(
                            -EDGE_SCRATCH_INIT_SCALE, EDGE_SCRATCH_INIT_SCALE))
                    for view in view_names
                })
        elif self.graph_mode == 'per_subject':
                self.k_per_view = k_per_view or {}
                self.graph_proj = nn.ModuleDict({
                    view: nn.Linear(n_subfeat_per_view[view], graph_proj_dim)
                    for view in view_names
                })
        elif self.graph_mode != 'static':
            raise ValueError(
                f"graph_mode must be 'static', 'learnable', 'learnable_scratch' or 'per_subject', got: {graph_mode!r}")
        # elif self.graph_mode != 'static':
        #     raise ValueError(
        #         f"graph_mode must be 'static', 'learnable', or 'learnable_scratch', got: {graph_mode!r}")

    def forward_features(self, view_inputs, extra=None, return_tokens=False):
        """view_inputs: dict view -> tensor (batch_size * n_nodes, n_subfeat),
        already on the target device."""
        view_embeddings = []
        for view in self.view_names:
            n_nodes = self.n_nodes_per_view[view]
            x = view_inputs[view]
            batch_size = x.shape[0] // n_nodes
            base_ei = self.base_edge_index[view]
            
            if self.graph_mode == 'per_subject':
                x_b = x.reshape(batch_size, n_nodes, -1)
                edge_index, edge_weight = build_per_subject_graph(
                    x_b, self.graph_proj[view], self.k_per_view.get(view, 16))

            elif self.graph_mode in LEARNABLE_GRAPH_MODES:
                edge_weight = self.edge_weight_params[view].repeat(batch_size)
                offsets = (torch.arange(batch_size, device=x.device) * n_nodes).view(-1, 1, 1)
                tiled = torch.as_tensor(base_ei, device=x.device).unsqueeze(0) + offsets
                edge_index = tiled.permute(1, 0, 2).reshape(2, -1)
            else:
                base_ew = self.base_edge_weight[view]
                edge_index_b, edge_weight_b = tile_graph_for_batch(base_ei, base_ew, n_nodes, batch_size)
                edge_index = torch.as_tensor(edge_index_b, device=x.device)
                edge_weight = torch.as_tensor(edge_weight_b, device=x.device)

            batch_vec = torch.arange(batch_size, device=x.device).repeat_interleave(n_nodes)

            # h = run_view_conv(self.view_convs[view], self.conv_type, x, edge_index, edge_weight)
            # h = self.relu(h)
            # h = self.dropout(h)
            h = x
            for i, conv in enumerate(self.view_convs[view]):
                h_new = run_view_conv(conv, self.conv_type, h, edge_index, edge_weight)
                # h_new = self.view_bns[view][i](h_new)
                h_new = self.relu(h_new)
                h_new = self.dropout(h_new)
                h = h_new + h if h.shape == h_new.shape else h_new   # residual از لایه دوم به بعد
            # h = tg.nn.global_mean_pool(h, batch_vec)
            if self.pool_type == 'attention':
                    h = self.att_pool[view](h, batch_vec)
            else:
                h = tg.nn.global_mean_pool(h, batch_vec)
            view_embeddings.append(h)
            
        if return_tokens:
            tokens = torch.stack(view_embeddings, dim=1)          # (B, 3, hid_c)
            if self.extra_dim > 0 and extra is not None:
                tokens = torch.cat([tokens, self.extra_mlp(extra).unsqueeze(1)], dim=1)  # (B, 4, hid_c)
            return tokens
        # fused = fuse_view_embeddings(self.fusion_type, self.fusion_module, view_embeddings)
        fused = fuse_view_embeddings(self.fusion_type, self.fusion_module, view_embeddings)
        if self.extra_dim > 0 and extra is not None:
                fused = torch.cat([fused, self.extra_mlp(extra)], dim=1)
        return fused
        # return fuse_view_embeddings(self.fusion_type, self.fusion_module, view_embeddings)

    def forward(self, view_inputs):
        return self.forward_features(view_inputs)
