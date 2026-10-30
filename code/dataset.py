import os
import torch
import csv
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import StratifiedKFold


def load_esm2_feature(seq_name, feature_dir):

    feature_path = os.path.join(feature_dir, f"{seq_name}.tensor")
    return torch.load(feature_path, map_location='cpu')


class PeptideDataset(Dataset):
    def __init__(self, csv_file, feature_dir, scaler=None, use_esm=True, max_len=190,
                 pca_extractor=None):

        self.data = []
        self.physico_features_raw = []
        self.physico_features_pca = []
        self.feature_dir = feature_dir
        self.scaler = scaler
        self.use_esm = use_esm
        self.max_len = max_len
        self.pca_extractor = pca_extractor

        self.aa_to_id = {aa: i + 1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}


        # 读取 CSV
        with open(csv_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row)
                seq = row['sequence']

                # 计算原始理化特征（588维）
                raw_feat = self._compute_raw_physico_features(seq)
                self.physico_features_raw.append(raw_feat)

                if pca_extractor is not None:
                    pca_feat = pca_extractor.get_features(seq)
                    self.physico_features_pca.append(pca_feat)

        print(f"数据集加载完成：{len(self.data)} 条样本")
        if pca_extractor is not None:
            print(f"  使用PCA降维，特征维度: {pca_extractor.feature_dim}")
        else:
            print(
                f"  使用原始理化特征，维度: {self.physico_features_raw[0].shape[0] if self.physico_features_raw else 0}")

    def _compute_raw_physico_features(self, seq):

        from physico_features import compute_aac, compute_dpc, compute_paac, compute_ctd
        import numpy as np

        aac = compute_aac(seq)
        dpc = compute_dpc(seq)
        paac = compute_paac(seq)
        ctd = compute_ctd(seq)

        return torch.tensor(np.concatenate([aac, dpc, paac, ctd]), dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def seq_to_tokens(self, sequence):
        sequence = ''.join(sequence.split())
        tokens = []
        for aa in sequence:
            if aa in self.aa_to_id:
                tokens.append(self.aa_to_id[aa])
            else:
                tokens.append(0)
        if len(tokens) < self.max_len:
            tokens = tokens + [0] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]
        return torch.tensor(tokens, dtype=torch.long)

    def __getitem__(self, idx):
        row = self.data[idx]
        name = row["name"]
        seq = row['sequence']

        label = torch.tensor(float(row['label']), dtype=torch.float)

        if self.use_esm:
            feature = load_esm2_feature(name, self.feature_dir)
        else:
            feature = torch.zeros(1, 1280)

        if self.pca_extractor is not None:
            physico_feat = self.physico_features_pca[idx].clone()
        else:
            physico_feat = self.physico_features_raw[idx].clone()


        if self.scaler is not None and self.pca_extractor is None:
            physico_feat = torch.tensor(
                self.scaler.transform([physico_feat.numpy()])[0],
                dtype=torch.float32
            )

        seq_tokens = self.seq_to_tokens(seq)
        seq_length = min(len(seq), self.max_len)

        return feature, label, seq, physico_feat, seq_tokens, seq_length


def collate_fn(batch):
    esm_features_list = []
    labels_list = []
    seqs_list = []
    esm_lengths = []
    physico_list = []
    seq_tokens_list = []
    seq_lengths_list = []

    for esm_feat, label, seq, physico_feat, seq_tokens, seq_len in batch:
        esm_features_list.append(esm_feat)
        labels_list.append(label)
        seqs_list.append(seq)
        esm_lengths.append(esm_feat.size(0))
        physico_list.append(physico_feat)
        seq_tokens_list.append(seq_tokens)
        seq_lengths_list.append(seq_len)

    padded_esm = pad_sequence(esm_features_list, batch_first=True, padding_value=0.0)
    labels_tensor = torch.stack(labels_list, dim=0)
    esm_lengths_tensor = torch.tensor(esm_lengths, dtype=torch.long)
    physico_tensor = torch.stack(physico_list, dim=0)

    seq_tokens_tensor = torch.stack(seq_tokens_list, dim=0)
    seq_lengths_tensor = torch.tensor(seq_lengths_list, dtype=torch.long)

    return padded_esm, labels_tensor, seqs_list, esm_lengths_tensor, physico_tensor, seq_tokens_tensor, seq_lengths_tensor


def kfold_split(dataset, n_splits=5, seed=42):

    indices = list(range(len(dataset)))
    labels = [float(dataset.data[i]["label"]) for i in indices]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for train_idx, val_idx in skf.split(indices, labels):
        yield train_idx, val_idx
