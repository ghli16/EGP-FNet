import torch
import torch.nn as nn

from peptide_gcn_encoder import PeptideGCNEncoder
from attention_modules import BiCrossAttention


# ==================== 工具函数 ====================
def masked_avg_pool(features, lengths):
    """带mask的均值池化"""
    B, T_max, D = features.shape
    mask = torch.arange(T_max, device=features.device)[None, :] < lengths[:, None]
    mask = mask.float().unsqueeze(-1)
    masked_sum = (features * mask).sum(dim=1)
    avg = masked_sum / mask.sum(dim=1).clamp(min=1e-8)
    return avg


# ==================== 拼接融合 ====================
class ConcatFusion(nn.Module):
    def __init__(self, num_representations, d_model=128):
        super().__init__()
        self.output_dim = num_representations * d_model

    def forward(self, tokens):
        B, N, D = tokens.shape
        concat = tokens.reshape(B, N * D)
        return concat, None


# ==================== 主模型 ====================
class PeptideModel(nn.Module):
    def __init__(self,
                 esm_dim: int = 1280,
                 physico_dim: int = 588,
                 encoder_type: str = "transformer",
                 pooling: str = "attn",
                 fusion_type: str = "concat",
                 use_esm: bool = True,
                 use_physico: bool = True,
                 use_peptide_gcn: bool = True,
                 use_gcn_aa_properties: bool = True,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_cross_layers: int = 2):
        super().__init__()

        self.pooling = pooling
        self.encoder_type = encoder_type
        self.use_esm = use_esm
        self.use_physico = use_physico
        self.use_peptide_gcn = use_peptide_gcn
        self.d_model = d_model

        self.use_cross_attention = use_esm and use_peptide_gcn

        # ===== GCN编码器 =====
        if use_peptide_gcn:
            self.peptide_gcn_encoder = PeptideGCNEncoder(
                use_onehot=True,
                use_aa_properties=use_gcn_aa_properties,
                hidden_dim=16,
                out_dim=d_model,
                dropout=0.1
            )

        # ===== ESM =====
        if use_esm:
            self.esm_proj = nn.Sequential(
                nn.Linear(esm_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, d_model)
            )
            self.esm_attn_pool = nn.Linear(d_model, 1)

        # ===== 理化特征 =====
        if use_physico:
            self.physico_proj = nn.Sequential(
                nn.Linear(physico_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, d_model)
            )

        # ===== 交叉注意力层 =====
        if self.use_cross_attention:
            self.cross_attention_layers = nn.ModuleList([
                BiCrossAttention(d_model, nhead, dropout=0.1)
                for _ in range(num_cross_layers)
            ])

        # ===== 计算融合的表示数量 =====
        self.num_representations = 0
        self.repr_names = []

        if use_esm:
            self.num_representations += 1
            self.repr_names.append('esm_pooled')
            if self.use_cross_attention:
                self.num_representations += 1
                self.repr_names.append('esm_cross_enhanced')

        if use_peptide_gcn:
            self.num_representations += 1
            self.repr_names.append('gcn_pooled')
            if self.use_cross_attention:
                self.num_representations += 1
                self.repr_names.append('gcn_cross_enhanced')

        if use_physico:
            self.num_representations += 1
            self.repr_names.append('physico')

        # ===== 拼接融合 =====
        if self.num_representations > 1:
            self.fusion = ConcatFusion(self.num_representations, d_model)
            classifier_input_dim = self.num_representations * d_model  # 640
        else:
            self.fusion = None
            classifier_input_dim = d_model

        # =====  MLP  =====
        if self.num_representations > 0:
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, 512),
                nn.BatchNorm1d(512),
                nn.GELU(),
                nn.Dropout(0.4),

                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(0.4),

                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.Dropout(0.3),

                nn.Linear(128, 1)
            )


        print(f"\nPeptideModel 初始化完成:")
        print(f"  - d_model: {d_model}")
        print(f"  - 使用ESM: {use_esm}")
        print(f"  - 使用理化特征: {use_physico}")
        print(f"  - 使用GCN: {use_peptide_gcn}")
        print(f"  - 交叉注意力: {self.use_cross_attention} (层数: {num_cross_layers if self.use_cross_attention else 0})")
        print(f"  - 融合方式: 拼接（{classifier_input_dim}维直接送入MLP）")
        print(f"  - MLP: {classifier_input_dim}-512-256-128-1 (GELU, Dropout 0.4/0.4/0.3)")
        print(f"  - 融合表示数量: {self.num_representations}")
        print(f"  - 表示类型: {self.repr_names}")

    # ==================== 池化方法 ====================
    def attention_pool(self, x, mask):
        scores = self.esm_attn_pool(x).squeeze(-1)
        scores = scores.masked_fill(mask, float('-inf'))
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights

    # ==================== 前向传播 ====================
    def forward(self, x, esm_lengths, physico_features, seq_tokens=None, seq_lengths=None):
        representations = []
        device = x.device if self.use_esm else physico_features.device

        if self.use_esm:
            B, L_esm = x.shape[0], x.shape[1]
            esm_mask = torch.arange(L_esm, device=device)[None, :] >= esm_lengths[:, None]
        else:
            B = physico_features.shape[0]
            esm_mask = None

        esm_tokens = None
        esm_pooled = None
        if self.use_esm:
            esm_proj = self.esm_proj(x)
            esm_tokens = esm_proj
            esm_pooled, _ = self.attention_pool(esm_proj, esm_mask)

        gcn_tokens = None
        gcn_pooled = None
        gcn_mask = None
        if self.use_peptide_gcn and seq_tokens is not None:
            gcn_node_features, gcn_pooled = self.peptide_gcn_encoder(
                seq_tokens, seq_lengths, return_nodes=True
            )
            gcn_tokens = gcn_node_features
            L_gcn = gcn_node_features.shape[1]
            gcn_mask = torch.arange(L_gcn, device=device)[None, :] >= seq_lengths[:, None]

        esm_cross_enhanced = None
        gcn_cross_enhanced = None
        if self.use_cross_attention and esm_tokens is not None and gcn_tokens is not None:
            esm_enhanced = esm_tokens
            gcn_enhanced = gcn_tokens
            for cross_attn in self.cross_attention_layers:
                esm_enhanced, gcn_enhanced = cross_attn(
                    esm_enhanced, esm_mask,
                    gcn_enhanced, gcn_mask
                )
            esm_cross_enhanced, _ = self.attention_pool(esm_enhanced, esm_mask)
            gcn_cross_enhanced = masked_avg_pool(gcn_enhanced, seq_lengths)

        if self.use_esm:
            if esm_cross_enhanced is not None:
                representations.append(esm_cross_enhanced)
            if esm_pooled is not None:
                representations.append(esm_pooled)

        if self.use_peptide_gcn:
            if gcn_cross_enhanced is not None:
                representations.append(gcn_cross_enhanced)
            if gcn_pooled is not None:
                representations.append(gcn_pooled)

        if self.use_physico:
            physico_proj = self.physico_proj(physico_features)
            representations.append(physico_proj)

        if len(representations) == 0:
            raise ValueError("没有可用的特征表示！")

        if len(representations) == 1:
            fused = representations[0]
            attn_weights = None
        else:
            stacked_repr = torch.stack(representations, dim=1)
            fused, attn_weights = self.fusion(stacked_repr)

        logits = self.classifier(fused).squeeze(-1)

        return logits, attn_weights

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))
