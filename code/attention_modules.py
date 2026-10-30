import torch
import torch.nn as nn

class CrossAttention(nn.Module):

    def __init__(self, d_model: int = 128, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) 必须能被 nhead ({nhead}) 整除"
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None):

        attn_out, _ = self.cross_attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        # 残差连接 + 归一化
        out = self.norm(query + self.dropout(attn_out))
        return out


class BiCrossAttention(nn.Module):
    """双向交叉注意力层：ESM-GCN 互相关注"""
    def __init__(self, d_model: int = 128, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.esm2gcn = CrossAttention(d_model, nhead, dropout)  # ESM 关注 GCN
        self.gcn2esm = CrossAttention(d_model, nhead, dropout)  # GCN 关注 ESM

    def forward(self, esm_tokens, esm_mask, gcn_tokens, gcn_mask):

        esm_enhanced = self.esm2gcn(
            query=esm_tokens,
            key=gcn_tokens,
            value=gcn_tokens,
            key_padding_mask=gcn_mask
        )

        gcn_enhanced = self.gcn2esm(
            query=gcn_tokens,
            key=esm_tokens,
            value=esm_tokens,
            key_padding_mask=esm_mask
        )

        return esm_enhanced, gcn_enhanced


class AttnFusion(nn.Module):
    def __init__(self, d_model: int = 128, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.score = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        U = torch.tanh(self.proj(tokens))

        scores = self.score(U).squeeze(-1)
        weights = torch.softmax(scores, dim=1)

        fused = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        fused = self.dropout(self.norm(fused))

        return fused, weights
