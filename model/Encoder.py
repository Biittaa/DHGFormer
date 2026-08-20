import torch.nn
import torch



class FullyConnectedOutput(torch.nn.Module):
    def __init__(self, embed_dim, input_dim):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 32),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(32, embed_dim),
            torch.nn.LeakyReLU(negative_slope=0.2),
            torch.nn.Dropout(p=0.1)
        )

        self.norm = torch.nn.LayerNorm(normalized_shape=embed_dim, elementwise_affine=True)

    def forward(self, x):

        x = self.norm(x)
        # [b, 50, 32] -> [b, 50, 32]
        out = self.fc(x)

        return out


def attention(Q, K, V, mask=None):
    l = Q.shape[2]
    num_head = Q.shape[1]
    score = torch.matmul(Q, K.permute(0, 1, 3, 2))
    score /= (Q.shape[-1] ** 0.5)

    if mask is not None:
        mask = torch.abs(mask)
        mask = mask.unsqueeze(1)
        mask = mask.expand(-1, 4, -1, -1)
        score = score * mask

    score = torch.softmax(score, dim=-1)
    x = torch.matmul(score, V)

    x = x.permute(0, 2, 1, 3).reshape(-1, l, num_head * Q.shape[3])
    return x


class MultiHead(torch.nn.Module):
    def __init__(self, input_dim, num_head, embed_dim):
        super().__init__()
        self.fc_Q = torch.nn.Linear(input_dim, 32)
        self.fc_K = torch.nn.Linear(input_dim, 32)
        self.fc_V = torch.nn.Linear(input_dim, 32)

        self.num_head = num_head

        self.out_fc = torch.nn.Linear(32, embed_dim)

        self.norm = torch.nn.LayerNorm(normalized_shape=input_dim, elementwise_affine=True)
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, Q, K, V, mask=None):

        # Q, K, V = [b, 50, 32]
        b = Q.shape[0]
        len = Q.shape[1]

        Q = self.norm(Q)
        K = self.norm(K)
        V = self.norm(V)

        K = self.fc_K(K)
        V = self.fc_V(V)
        Q = self.fc_Q(Q)

        Q = Q.reshape(b, len, self.num_head, -1).permute(0, 2, 1, 3)
        K = K.reshape(b, len, self.num_head, -1).permute(0, 2, 1, 3)
        V = V.reshape(b, len, self.num_head, -1).permute(0, 2, 1, 3)

        score = attention(Q, K, V, mask)
        score = self.dropout(self.out_fc(score))

        return score


class EncoderLayer(torch.nn.Module):
    def __init__(self, input_dim, num_head, embed_dim):
        super(EncoderLayer, self).__init__()
        self.mh = MultiHead(input_dim, num_head, embed_dim)
        self.fc = FullyConnectedOutput(embed_dim, input_dim)

    def forward(self, x, mask=None):
        score = self.mh(x, x, x, mask)
        out = self.fc(score)

        return out


class FCEncoder(torch.nn.Module):
    def __init__(self, input_dim, num_head, embed_dim):
        super(FCEncoder, self).__init__()
        self.layer = EncoderLayer(input_dim, num_head, embed_dim)

    def forward(self, x, mask=None):
        x = self.layer(x, mask)

        return x
    
    
class SMRIFCNEncoder(torch.nn.Module):
    """FCN-style sMRI encoder. Mirrors FCN.ipynb's first two linear blocks;
    output is the embedding BEFORE the final classification layer, so it can
    be concatenated with the fMRI embedding for late fusion."""

    def __init__(self, input_dim, hid_1=500, hid_2=30, dropout=0.5):
        super().__init__()
        self.linear_1 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hid_1),
            torch.nn.Dropout(dropout),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm1d(hid_1)
        )
        self.linear_2 = torch.nn.Linear(hid_1, hid_2)
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.relu(self.linear_1(x))
        x = self.relu(self.dropout(self.linear_2(x)))
        return x
    
import torch.nn as nn   
class ModalityAttentionFusion(nn.Module):
    """Learns per-sample importance weights for fMRI vs sMRI embeddings
    before fusing them (weighted concat)."""

    def __init__(self, fmri_dim, smri_dim, hidden_dim=64):
        super().__init__()
        self.fmri_proj = nn.Linear(fmri_dim, hidden_dim)
        self.smri_proj = nn.Linear(smri_dim, hidden_dim)
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)   
        )

    def forward(self, fmri_emb, smri_emb):
        f = self.fmri_proj(fmri_emb)
        s = self.smri_proj(smri_emb)
        scores = self.attn(torch.cat([f, s], dim=1))       # [B, 2]
        weights = torch.softmax(scores, dim=1)             
        w_f, w_s = weights[:, 0:1], weights[:, 1:2]

        fused = torch.cat([w_f * fmri_emb, w_s * smri_emb], dim=1)
        return fused, weights     
    
    
    
    
    
    
class SMRITransformerEncoder(torch.nn.Module):
    """Lightweight Transformer sMRI encoder. Splits the raw feature vector
    into small patches (tokens) instead of treating every feature as its
    own token, so attention stays cheap even without Ridge-RFE feature
    selection. Output dim = hid_2, so it drops into fusion_classifier
    unchanged."""

    def __init__(self, input_dim, patch_size=32, embed_dim=64,
                 num_heads=4, num_layers=2, hid_2=30, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (input_dim + patch_size - 1) // patch_size
        self.pad_len = self.num_patches * patch_size - input_dim

        self.patch_proj = torch.nn.Linear(patch_size, embed_dim)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = torch.nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim))
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 2, dropout=dropout,
            batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        self.out_proj = torch.nn.Linear(embed_dim, hid_2)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        b = x.shape[0]
        if self.pad_len > 0:
            pad = torch.zeros(b, self.pad_len, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)

        patches = x.view(b, self.num_patches, self.patch_size)
        tokens = self.patch_proj(patches)

        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed

        out = self.transformer(tokens)
        cls_out = out[:, 0, :]
        return self.dropout(self.out_proj(cls_out))
