import numpy as np
import torch
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

HYDROPHOBICITY = {
    'A': 0.62, 'C': 0.29, 'D': -0.9, 'E': -0.74, 'F': 1.19, 'G': 0.48,
    'H': -0.4, 'I': 1.38, 'K': -1.5, 'L': 1.06, 'M': 0.64, 'N': -0.78,
    'P': 0.12, 'Q': -0.85, 'R': -2.53, 'S': -0.18, 'T': -0.05, 'V': 1.08,
    'W': 0.81, 'Y': 0.26,
}


def _clean_sequence(seq):

    seq = seq.strip().upper()
    cleaned = []
    valid = []
    unknown_positions = []

    for i, aa in enumerate(seq):
        if aa in AMINO_ACIDS:
            cleaned.append(aa)
            valid.append(aa)
        else:
            cleaned.append('X')
            unknown_positions.append(i)

    return ''.join(cleaned), ''.join(valid), unknown_positions


def compute_aac(seq):
    """氨基酸组成 (20维)
    """
    _, valid_seq, _ = _clean_sequence(seq)

    if len(valid_seq) == 0:
        return np.zeros(20)

    counter = Counter(valid_seq)
    total = len(valid_seq)

    feat = []
    for aa in AMINO_ACIDS:
        feat.append(counter.get(aa, 0) / (total + 1e-6))

    return np.array(feat)


def compute_dpc(seq):
    """二肽组成 (400维)
    """
    cleaned_seq, _, unknown_positions = _clean_sequence(seq)
    L = len(cleaned_seq)

    if L < 2:
        return np.zeros(400)

    unknown_set = set(unknown_positions)
    aa_index = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    mat = np.zeros((20, 20))
    valid_pairs = 0

    for i in range(L - 1):
        if i in unknown_set or (i + 1) in unknown_set:
            continue

        a1 = cleaned_seq[i]
        a2 = cleaned_seq[i + 1]

        if a1 in AMINO_ACIDS and a2 in AMINO_ACIDS:
            mat[aa_index[a1]][aa_index[a2]] += 1
            valid_pairs += 1

    if valid_pairs > 0:
        mat = mat / valid_pairs

    return mat.flatten()


def compute_paac(seq, lambda_value=1, w=0.05):
    """伪氨基酸组成 (21维)
    """
    cleaned_seq, _, unknown_positions = _clean_sequence(seq)
    unknown_set = set(unknown_positions)
    L = len(cleaned_seq)

    if L < 2:
        return np.zeros(21)

    aac = compute_aac(seq)
    theta = []

    for n in range(1, lambda_value + 1):
        corr = 0.0
        count = 0

        for i in range(L - n):
            if i in unknown_set or (i + n) in unknown_set:
                continue

            a1 = cleaned_seq[i]
            a2 = cleaned_seq[i + n]

            if a1 in HYDROPHOBICITY and a2 in HYDROPHOBICITY:
                corr += (HYDROPHOBICITY[a1] - HYDROPHOBICITY[a2]) ** 2
                count += 1

        theta.append(corr / (count + 1e-6))

    theta = np.array(theta)
    if len(theta) > 0 and np.std(theta) > 0:
        theta = theta / (np.std(theta) + 1e-6)

    denom = np.sum(aac) + w * np.sum(theta)
    if denom == 0:
        return np.zeros(21)

    paac = []
    for f in aac:
        paac.append(f / denom)
    for t in theta:
        paac.append((w * t) / denom)

    return np.array(paac)


CTD_PROPERTIES = {
    "hydrophobicity": ["RKEDQN", "GASTPHY", "CLVIMFW"],
    "normwaalsvolume": ["GASTPDC", "NVEQIL", "MHKFRYW"],
    "polarity": ["LIFWCMVY", "PATGS", "HQRKNED"],
    "polarizability": ["GASDT", "CPNVEQIL", "KMHFRYW"],
    "charge": ["KR", "ANCQGHILMFPSTWYV", "DE"],
    "secondarystruct": ["EALMQKRH", "VIYCWFT", "GNPSD"],
    "solventaccess": ["ALFCGIVW", "RKQEND", "MPSTHY"]
}


def _create_group_map(groups):
    group_map = {}
    for i, g in enumerate(groups):
        for aa in g:
            group_map[aa] = i + 1
    return group_map


def _group_sequence(seq, groups):
    group_map = _create_group_map(groups)
    gseq = []
    valid_positions = []

    for i, aa in enumerate(seq):
        if aa == 'X':
            gseq.append(None)
        elif aa in group_map:
            gseq.append(group_map[aa])
            valid_positions.append(i)
        else:
            gseq.append(None)

    return gseq, valid_positions


def _composition(gseq, valid_positions):
    """CTD组成特征
    """
    L = len(valid_positions)
    if L == 0:
        return [0.0, 0.0, 0.0]

    counts = {1: 0, 2: 0, 3: 0}
    for pos in valid_positions:
        g = gseq[pos]
        if g in counts:
            counts[g] += 1

    return [
        counts[1] / (L + 1e-6),
        counts[2] / (L + 1e-6),
        counts[3] / (L + 1e-6),
    ]


def _transition(gseq, valid_positions):

    if len(valid_positions) < 2:
        return [0.0, 0.0, 0.0]

    c12 = c13 = c23 = 0
    total_transitions = 0

    for idx in range(len(valid_positions) - 1):
        pos1 = valid_positions[idx]
        pos2 = valid_positions[idx + 1]

        g1 = gseq[pos1]
        g2 = gseq[pos2]

        if g1 and g2 and g1 != g2:
            pair = {g1, g2}
            if pair == {1, 2}:
                c12 += 1
            elif pair == {1, 3}:
                c13 += 1
            elif pair == {2, 3}:
                c23 += 1
            total_transitions += 1

    if total_transitions == 0:
        return [0.0, 0.0, 0.0]

    return [
        c12 / (total_transitions + 1e-6),
        c13 / (total_transitions + 1e-6),
        c23 / (total_transitions + 1e-6),
    ]


def _distribution(gseq, valid_positions):
    """CTD分布特征
    """
    result = []
    L = len(gseq)

    if len(valid_positions) == 0:
        return [0.0] * 15

    group_positions = {1: [], 2: [], 3: []}
    for pos in valid_positions:
        g = gseq[pos]
        if g in group_positions:
            group_positions[g].append(pos + 1)

    percentiles = [0, 0.25, 0.5, 0.75, 0.99]

    for g in [1, 2, 3]:
        positions = group_positions[g]

        if len(positions) == 0:
            result += [0.0] * 5
        else:
            for p in percentiles:
                idx = int(len(positions) * p)
                idx = min(idx, len(positions) - 1)
                result.append(positions[idx] / L)

    return result


def compute_ctd(seq):
    """CTD特征 (147维)
    """
    cleaned_seq, _, _ = _clean_sequence(seq)

    features = []

    for prop_name, groups in CTD_PROPERTIES.items():
        gseq, valid_positions = _group_sequence(cleaned_seq, groups)

        features += _composition(gseq, valid_positions)
        features += _transition(gseq, valid_positions)
        features += _distribution(gseq, valid_positions)

    return np.array(features)


class PhysicoFeatureExtractor:

    def __init__(self, pca_components=0.95):

        self.raw_feature_dim = 588
        self.pca_components = pca_components
        self.feature_dim = None
        self.pca = None
        self.scaler = None

    def fit(self, sequences):
        """在训练集序列上拟合PCA"""
        if isinstance(self.pca_components, float):
            print(f"正在拟合PCA，目标: 保留 {self.pca_components * 100}% 的方差信息...")
        else:
            print(f"正在拟合PCA，目标维度: {self.pca_components}")

        # 1. 提取所有序列的原始特征
        all_features = []
        for seq in sequences:
            feat = self._extract_raw_features(seq)
            all_features.append(feat)

        all_features = np.array(all_features)
        print(f"  原始特征矩阵形状: {all_features.shape}")

        # 2. 标准化
        self.scaler = StandardScaler()
        all_features_scaled = self.scaler.fit_transform(all_features)

        # 3. 拟合PCA
        self.pca = PCA(n_components=self.pca_components)
        self.pca.fit(all_features_scaled)

        self.feature_dim = self.pca.n_components_

        explained_var = np.sum(self.pca.explained_variance_ratio_)
        print(f"  PCA完成，{self.feature_dim} 个主成分解释了 "
              f"{explained_var:.4f} ({explained_var * 100:.2f}%) 的方差")

        return self

    def save(self, path):

        joblib.dump(
            {'pca': self.pca, 'scaler': self.scaler, 'feature_dim': self.feature_dim},
            path
        )
        print(f"  PCA模型已保存至: {path}")

    def load(self, path):
        model = joblib.load(path)
        self.pca = model['pca']
        self.scaler = model['scaler']
        self.feature_dim = model.get('feature_dim', self.pca.n_components_)
        self.pca_components = self.feature_dim
        print(f"  PCA模型已加载: {path}")
        print(f"  降维后维度: {self.feature_dim}")

    def _extract_raw_features(self, seq):
        seq = seq.strip().upper()
        if len(seq) == 0:
            return np.zeros(self.raw_feature_dim)

        aac = compute_aac(seq)
        dpc = compute_dpc(seq)
        paac = compute_paac(seq)
        ctd = compute_ctd(seq)

        return np.concatenate([aac, dpc, paac, ctd])

    def get_features(self, seq):

        if self.pca is None or self.scaler is None:
            raise RuntimeError(
                "PCA未初始化。先调用fit()方法拟合，或load()加载预训练模型"
            )

        raw_feat = self._extract_raw_features(seq)
        raw_feat_scaled = self.scaler.transform(raw_feat.reshape(1, -1))
        feat_pca = self.pca.transform(raw_feat_scaled)

        return torch.tensor(feat_pca[0], dtype=torch.float32)

    def get_raw_features(self, seq):

        return torch.tensor(self._extract_raw_features(seq), dtype=torch.float32)
