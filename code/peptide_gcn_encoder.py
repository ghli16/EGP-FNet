import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


class AminoAcidProperties:

    PROPERTIES = {
        'A': [0.62, 0.0, 89.09, 88.6, 6.00, 8.1],
        'R': [-2.53, 1.0, 174.20, 173.4, 10.76, 10.5],
        'N': [-0.78, 0.0, 132.12, 114.1, 5.41, 11.6],
        'D': [-0.90, -1.0, 133.10, 111.1, 2.77, 13.0],
        'C': [0.29, 0.0, 121.16, 108.5, 5.05, 5.5],
        'Q': [-0.85, 0.0, 146.15, 143.8, 5.65, 10.5],
        'E': [-0.74, -1.0, 147.13, 138.4, 3.22, 12.3],
        'G': [0.48, 0.0, 75.07, 60.1, 5.97, 9.0],
        'H': [-0.40, 0.5, 155.16, 153.2, 7.59, 10.4],
        'I': [1.38, 0.0, 131.18, 166.7, 6.02, 5.2],
        'L': [1.06, 0.0, 131.18, 166.7, 5.98, 4.9],
        'K': [-1.50, 1.0, 146.19, 168.6, 9.74, 11.3],
        'M': [0.64, 0.0, 149.21, 162.9, 5.74, 5.7],
        'F': [1.19, 0.0, 165.19, 189.9, 5.48, 5.2],
        'P': [0.12, 0.0, 115.13, 112.7, 6.30, 8.0],
        'S': [-0.18, 0.0, 105.09, 89.0, 5.68, 9.2],
        'T': [-0.05, 0.0, 119.12, 116.1, 5.60, 8.6],
        'W': [0.81, 0.0, 204.23, 227.8, 5.89, 5.4],
        'Y': [0.26, 0.0, 181.19, 193.6, 5.66, 6.2],
        'V': [1.08, 0.0, 117.15, 140.0, 5.96, 5.9],
    }

    AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
    ID_TO_AA = {i + 1: aa for i, aa in enumerate(AMINO_ACIDS)}

    @classmethod
    def property_dim(cls):
        return len(cls.PROPERTIES['A'])

    @staticmethod
    def get_properties_tensor(tokens, device):

        properties = []
        for token in tokens:
            token_val = token.item()
            if 1 <= token_val <= 20:
                aa = AminoAcidProperties.ID_TO_AA[token_val]
            else:
                aa = 'A'
            props = AminoAcidProperties.PROPERTIES[aa]
            properties.append(props)
        return torch.tensor(properties, device=device, dtype=torch.float)


class PeptideGCNEncoder(nn.Module):

    def __init__(self,
                 use_onehot=True,
                 use_aa_properties=True,
                 hidden_dim=16,
                 out_dim=128,
                 dropout=0.1):

        super().__init__()

        self.use_onehot = use_onehot
        self.use_aa_properties = use_aa_properties
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # 计算输入维度
        input_dim = 0
        if use_onehot:
            input_dim += 20
        if use_aa_properties:
            input_dim += AminoAcidProperties.property_dim()

        if input_dim == 0:
            raise ValueError("至少需要启用 onehot 或 aa_properties 中的一个")

        # 网络层
        self.node_proj = nn.Linear(input_dim, hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim * 2, out_dim)

        self.node_out_proj = nn.Linear(hidden_dim, out_dim)

        self.dropout = dropout

        print(f"PeptideGCNEncoder 初始化:")
        print(f"  - One-hot: {use_onehot} (20-dim)")
        print(f"  - 理化性质: {use_aa_properties} (6-dim)")
        print(f"  - 输入维度: {input_dim}")
        print(f"  - 隐藏维度: {hidden_dim}")
        print(f"  - 输出维度: {out_dim}")

    def _onehot_encode_20dim(self, tokens):

        N = len(tokens)
        device = tokens.device

        onehot = torch.zeros(N, 20, device=device)
        valid_mask = (tokens >= 1) & (tokens <= 20)
        onehot[valid_mask, tokens[valid_mask] - 1] = 1.0

        return onehot

    def _build_graph(self, seq_tokens, seq_lengths, device):
        """将批次序列转换为图结构
        """
        B, max_len = seq_tokens.shape
        all_node_features = []
        all_edge_indices = []
        all_batch = []
        total_nodes = 0

        for b in range(B):
            length = seq_lengths[b].item()
            if length == 0:
                continue

            # 提取实际长度的tokens
            tokens = seq_tokens[b, :length]

            # 构建节点特征
            features = []

            # One-hot编码（20维）
            if self.use_onehot:
                onehot = self._onehot_encode_20dim(tokens)
                features.append(onehot)

            # 理化性质（6维）
            if self.use_aa_properties:
                aa_props = []
                for token in tokens:
                    token_val = token.item()
                    if 1 <= token_val <= 20:
                        aa = AminoAcidProperties.ID_TO_AA[token_val]
                    else:
                        aa = 'A'
                    aa_props.append(AminoAcidProperties.PROPERTIES[aa])
                aa_props = torch.tensor(aa_props, device=device, dtype=torch.float)
                features.append(aa_props)

            node_feat = torch.cat(features, dim=-1)
            all_node_features.append(node_feat)

            # 构建边（序列相邻连接，双向）
            L = len(tokens)
            edges = []
            for i in range(L - 1):
                edges.append([i, i + 1])
                edges.append([i + 1, i])

            if edges:
                edge_index = torch.tensor(edges, device=device).t().contiguous()
                edge_index = edge_index + total_nodes
                all_edge_indices.append(edge_index)

            batch_idx = torch.full((L,), b, device=device, dtype=torch.long)
            all_batch.append(batch_idx)
            total_nodes += L

        if not all_node_features:
            return None, None, None, B

        node_features = torch.cat(all_node_features, dim=0)
        edge_index = torch.cat(all_edge_indices, dim=1) if all_edge_indices else \
            torch.tensor([[], []], device=device, dtype=torch.long)
        batch = torch.cat(all_batch, dim=0)

        return node_features, edge_index, batch, B

    def _reshape_nodes_to_batch(self, x, batch, B, max_len):

        device = x.device
        D = x.size(-1)
        out = torch.zeros(B, max_len, D, device=device)

        for b in range(B):
            mask = batch == b
            nodes_b = x[mask]  # [L_b, D]
            n = min(nodes_b.size(0), max_len)
            out[b, :n] = nodes_b[:n]

        return out

    def forward(self, seq_tokens, seq_lengths, return_nodes=False):

        device = seq_tokens.device
        B = seq_tokens.shape[0]
        max_len = seq_tokens.shape[1]

        # 1. 构建图
        node_features, edge_index, batch, B_actual = self._build_graph(
            seq_tokens, seq_lengths, device
        )

        # 2. 处理空批次
        if node_features is None:
            if return_nodes:
                return (
                    torch.zeros(B, max_len, self.out_dim, device=device),
                    torch.zeros(B, self.out_dim, device=device)
                )
            return torch.zeros(B, self.out_dim, device=device)

        # 3. 节点特征投影
        x = self.node_proj(node_features)
        residual = x


        # 4. 三层图卷积（消息传递）
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)

        x = x + residual


        # 保存池化前的节点特征
        node_features_before_pool = x

        # 5. 全局池化
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        pooled = torch.cat((mean_pool, max_pool), dim=-1)


        pooled = F.dropout(pooled, p=self.dropout, training=self.training)
        pooled = self.out_proj(pooled)

        if return_nodes:
            node_features_batch = self._reshape_nodes_to_batch(
                node_features_before_pool, batch, B_actual, max_len
            )
            node_features_batch = self.node_out_proj(node_features_batch)
            return node_features_batch, pooled

        return pooled
