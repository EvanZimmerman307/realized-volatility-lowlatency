import torch, torch.nn as nn

class TinyRVTransformer(nn.Module):
    def __init__(self, in_dim, d_model=64, nhead=2, nlayers=2, dim_ff=128, dropout=0.1, seq_len=600):
        super().__init__()
        self.seq_len = seq_len
        self.proj = nn.Linear(in_dim, d_model)

        # Learnable positional embeddings (+1 for CLS)
        self.pos = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, nlayers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x):                    # x: [B, 600, F]
        h = self.proj(x)                     # [B, 600, d]
        cls = self.cls.expand(h.size(0), -1, -1)    # [B,1,d]
        h = torch.cat([cls, h], dim=1)              # [B, 601, d]
        h = h + self.pos[:, : h.size(1), :]         # add positions
        h = self.enc(h)                             
        cls_out = h[:, 0]                           # [B, d]
        return self.head(cls_out).squeeze(-1)
