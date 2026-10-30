import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from model import PeptideModel
from dataset import PeptideDataset, collate_fn
from physico_features import PhysicoFeatureExtractor
import numpy as np
from sklearn.metrics import (roc_auc_score, f1_score, recall_score, precision_score,
                             matthews_corrcoef, accuracy_score, average_precision_score)
from sklearn.model_selection import train_test_split

import warnings
import json
import time

warnings.filterwarnings("ignore")

# ==================== 随机种子 ====================
seed = 42


def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True, warn_only=True)
        elif hasattr(torch, 'set_deterministic'):
            torch.set_deterministic(True)

    try:
        import torch_geometric
        torch_geometric.seed_everything(seed)
    except ImportError:
        pass


set_seed(seed)


# ==================== 评估指标 ====================
def compute_binary_metrics(y_true, y_pred, y_prob):
    """计算二分类评估指标"""
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()

    y_pred_bin = y_pred

    try:
        pr = recall_score(y_true, y_pred_bin, pos_label=1, zero_division=0)
        nr = recall_score(y_true, y_pred_bin, pos_label=0, zero_division=0)
        precision = precision_score(y_true, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true, y_pred_bin, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred_bin)
    except Exception:
        pr, nr, precision, f1, mcc = 0.0, 0.0, 0.0, 0.0, 0.0

    accuracy = accuracy_score(y_true, y_pred_bin)

    try:
        ap = average_precision_score(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        ap, auc = 0.0, 0.0

    return {
        'PR': pr, 'NR': nr, 'precision': precision, 'accuracy': accuracy,
        'AP': ap, 'F1': f1, 'MCC': mcc, 'AUC': auc
    }


def save_metrics_log(metrics, log_path):
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"日志已保存至：{log_path}")


def train_model(model, train_loader, val_loader, device, epochs=50, patience=10, lr=5e-5):
    """训练模型，使用验证集早停
    """
    y_all = []
    for batch in train_loader:
        _, y, _, _, _, _, _ = batch
        y_all.extend(y.numpy())
    y_all = np.array(y_all)

    if len(np.unique(y_all)) > 1:
        n_pos = np.sum(y_all == 1)
        n_neg = np.sum(y_all == 0)
        pos_weight_val = n_neg / n_pos
        print(f"训练集正负样本比: {n_pos}/{n_neg} = {n_pos / n_neg:.2f}:1")
        print(f"pos_weight: {pos_weight_val:.3f}")
        pos_weight = torch.tensor([pos_weight_val], device=device)
    else:
        pos_weight = torch.tensor([1.0], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_score = -999
    best_state = None
    best_metrics = None
    early_stop_counter = 0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        # ===== 训练 =====
        model.train()
        total_loss = 0
        for batch in train_loader:
            esmFeatures, y, _, esm_lengths, physico, seq_tokens, seq_lengths = batch

            esmFeatures = esmFeatures.to(device)
            y = y.to(device)
            esm_lengths = esm_lengths.to(device)
            physico = physico.to(device)
            seq_tokens = seq_tokens.to(device)
            seq_lengths = seq_lengths.to(device)

            out, attn_weights = model(
                esmFeatures, esm_lengths, physico,
                seq_tokens, seq_lengths
            )
            loss = criterion(out.squeeze(), y.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ===== 验证 =====
        model.eval()
        y_true, y_prob = [], []

        with torch.no_grad():
            for batch in val_loader:
                esmFeatures, y, _, esm_lengths, physico, seq_tokens, seq_lengths = batch

                esmFeatures = esmFeatures.to(device)
                y = y.to(device)
                esm_lengths = esm_lengths.to(device)
                physico = physico.to(device)
                seq_tokens = seq_tokens.to(device)
                seq_lengths = seq_lengths.to(device)

                out, _ = model(
                    esmFeatures, esm_lengths, physico,
                    seq_tokens, seq_lengths
                )
                prob = torch.sigmoid(out).cpu().numpy()

                y_true.extend(y.cpu().numpy())
                y_prob.extend(prob)

        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        y_pred = (y_prob > 0.5).astype(int)
        metrics = compute_binary_metrics(y_true, y_pred, y_prob)
        score = metrics['MCC']

        print(
            f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | Val ACC: {metrics['accuracy']:.4f} | Val MCC: {metrics['MCC']:.4f}")

        if score > best_score:
            best_score = score
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_metrics = metrics
            early_stop_counter = 0
            best_epoch = epoch
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

    print(f"\n最佳 Epoch: {best_epoch} | 最佳 Val MCC: {best_score:.4f}")
    print(
        f"最佳验证集指标: ACC={best_metrics['accuracy']:.4f}, AUC={best_metrics['AUC']:.4f}, F1={best_metrics['F1']:.4f}")

    return best_state, best_metrics


def evaluate(model, test_loader, device):
    """在测试集上评估模型"""
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for batch in test_loader:
            esmFeatures, y, _, esm_lengths, physico, seq_tokens, seq_lengths = batch

            esmFeatures = esmFeatures.to(device)
            y = y.to(device)
            esm_lengths = esm_lengths.to(device)
            physico = physico.to(device)
            seq_tokens = seq_tokens.to(device)
            seq_lengths = seq_lengths.to(device)

            out, _ = model(
                esmFeatures, esm_lengths, physico,
                seq_tokens, seq_lengths
            )
            prob = torch.sigmoid(out).cpu().numpy()
            y_true.extend(y.cpu().numpy())
            y_prob.extend(prob)

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob > 0.5).astype(int)
    metrics = compute_binary_metrics(y_true, y_pred, y_prob)
    return metrics


# ==================== 主函数 ====================
def main():
    set_seed(seed)
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', required=True, help='训练集CSV文件')
    parser.add_argument('--feature_dir', required=True, help='ESM2特征目录')
    parser.add_argument('--test_csv', required=True, help='测试集CSV文件')
    parser.add_argument('--log_path', default='log/final_metrics.json', help='日志保存路径')
    parser.add_argument('--model_dir', default='models_binary', help='模型保存目录')

    parser.add_argument('--use_esm', action='store_true', default=False, help='使用ESM2特征')
    parser.add_argument('--use_physico', action='store_true', default=False, help='使用理化特征')
    parser.add_argument('--use_peptide_gcn', action='store_true', default=False, help='使用GCN编码器')
    parser.add_argument('--use_gcn_aa_properties', action='store_true', default=False,
                        help='GCN节点是否使用氨基酸理化性质')

    parser.add_argument('--d_model', type=int, default=128,
                        help='统一特征维度（ESM/GCN/Physico都投影到这个维度）')
    parser.add_argument('--nhead', type=int, default=4,
                        help='交叉注意力的头数')
    parser.add_argument('--num_cross_layers', type=int, default=2,
                        help='交叉注意力层数')

    parser.add_argument('--lr', type=float, default=5e-5, help='学习率')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='最大训练轮数')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--max_len', type=int, default=190, help='最大序列长度')

    parser.add_argument('--pca_components', type=float, default=0.95,
                        help='PCA降维参数')

    args = parser.parse_args()

    device = torch.device('cpu')
    os.makedirs(args.model_dir, exist_ok=True)

    print("=" * 60)
    print("模型训练")
    print("=" * 60)
    print(f"设备：{device}")
    print(f"\n特征配置：")
    print(f"  - ESM2: {args.use_esm}")
    print(f"  - 理化特征: {args.use_physico} (使用PCA降维)")
    print(f"  - PeptideGCN: {args.use_peptide_gcn}")
    print(f"  - GCN氨基酸理化性质: {args.use_gcn_aa_properties}")
    if args.use_esm and args.use_peptide_gcn:
        print(f"  交叉注意力: ({args.num_cross_layers}层, {args.nhead}头)")

    print(f"\n模型架构参数:")
    print(f"  - d_model: {args.d_model}")
    print(f"  - nhead: {args.nhead}")
    print(f"  - num_cross_layers: {args.num_cross_layers}")
    print(f"\n超参数：")
    print(f"  - 学习率: {args.lr}")
    print(f"  - 批次大小: {args.batch_size}")
    print(f"  - 最大轮数: {args.epochs}")
    print(f"  - 早停耐心: {args.patience}")
    print(f"  - 验证集比例: {args.val_ratio}")
    print(f"  - PCA参数: {args.pca_components}")
    print(f"  - 最大序列长度 (max_len): {args.max_len}")

    # ==================== 1. 加载训练集 ====================
    print("\n=== 步骤1：加载训练集 ===")
    train_dataset_full = PeptideDataset(
        args.train_csv,
        args.feature_dir,
        scaler=None,
        use_esm=args.use_esm,
        pca_extractor=None,
        max_len=args.max_len
    )
    print(f"训练集总样本数：{len(train_dataset_full)}")

    # ==================== 2. 划分训练集和验证集 ====================
    print("\n=== 步骤2：划分训练集和验证集 ===")
    all_indices = list(range(len(train_dataset_full)))
    all_labels = [float(train_dataset_full.data[i]["label"]) for i in all_indices]

    train_idx, val_idx = train_test_split(
        all_indices,
        test_size=args.val_ratio,
        random_state=seed,
        stratify=all_labels
    )
    print(f"训练集样本数：{len(train_idx)}")
    print(f"验证集样本数：{len(val_idx)}")

    # ==================== 3. 拟合PCA ====================
    print("\n=== 步骤3：拟合PCA ===")

    pca_extractor = PhysicoFeatureExtractor(pca_components=args.pca_components)
    train_sequences = [train_dataset_full.data[i]["sequence"] for i in train_idx]
    print(f"用于拟合PCA的序列数: {len(train_sequences)}")

    pca_extractor.fit(train_sequences)
    physico_dim = pca_extractor.feature_dim
    print(f"\n*** 理化特征降维后维度: {physico_dim} ***")

    pca_model_path = os.path.join(args.model_dir, 'physico_pca.joblib')
    pca_extractor.save(pca_model_path)

    # ==================== 4. 创建带PCA的数据集 ====================
    print("\n=== 步骤4：创建带PCA的数据集 ===")

    train_dataset_pca = PeptideDataset(
        args.train_csv,
        args.feature_dir,
        scaler=None,
        use_esm=args.use_esm,
        pca_extractor=pca_extractor,
        max_len=args.max_len
    )

    train_set = Subset(train_dataset_pca, train_idx)
    val_set = Subset(train_dataset_pca, val_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        generator=torch.Generator().manual_seed(seed)
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        generator=torch.Generator().manual_seed(seed)
    )

    # ==================== 5. 创建模型 ====================
    print("\n=== 步骤5：创建模型 ===")
    model = PeptideModel(
        physico_dim=physico_dim,
        encoder_type="transformer",
        pooling="attn",
        fusion_type="concat",
        use_esm=args.use_esm,
        use_physico=args.use_physico,
        use_peptide_gcn=args.use_peptide_gcn,
        use_gcn_aa_properties=args.use_gcn_aa_properties,
        d_model=args.d_model,
        nhead=args.nhead,
        num_cross_layers=args.num_cross_layers
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量：总参数 {total_params:,}，可训练参数 {trainable_params:,}")

    # ==================== 6. 训练 ====================
    print("\n=== 步骤6：开始训练 ===")
    best_state, val_metrics = train_model(
        model, train_loader, val_loader, device,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr
    )

    # ==================== 7. 保存模型 ====================
    print("\n=== 步骤7：保存模型 ===")
    final_model_path = os.path.join(args.model_dir, 'final_model.pt')
    torch.save({
        'model_state_dict': best_state,
        'pca_model_path': pca_model_path,
        'physico_dim': physico_dim,
        'val_metrics': val_metrics,
        'args': vars(args)
    }, final_model_path)
    print(f"模型已保存至：{final_model_path}")

    # ==================== 8. 测试集评估 ====================
    print("\n=== 步骤8：加载测试集并评估 ===")

    test_pca_extractor = PhysicoFeatureExtractor(pca_components=args.pca_components)
    test_pca_extractor.load(pca_model_path)

    test_dataset = PeptideDataset(
        args.test_csv,
        args.feature_dir,
        scaler=None,
        use_esm=args.use_esm,
        pca_extractor=test_pca_extractor,
        max_len=args.max_len
    )
    print(f"测试集样本数：{len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        generator=torch.Generator().manual_seed(seed)
    )

    print("\n=== 测试集评估 ===")
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)

    print("\n测试集结果：")
    print(f"  - Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  - Precision: {test_metrics['precision']:.4f}")
    print(f"  - Recall (PR): {test_metrics['PR']:.4f}")
    print(f"  - Specificity (NR): {test_metrics['NR']:.4f}")
    print(f"  - MCC: {test_metrics['MCC']:.4f}")
    print(f"  - F1 Score: {test_metrics['F1']:.4f}")
    print(f"  - AUC: {test_metrics['AUC']:.4f}")
    print(f"  - AP: {test_metrics['AP']:.4f}")


    # ==================== 9. 保存日志 ====================
    final_log = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'mode': 'final_training_with_cross_attention',
        'pca_components': args.pca_components,
        'physico_dim_after_pca': physico_dim,
        'model_config': {
            'd_model': args.d_model,
            'nhead': args.nhead,
            'num_cross_layers': args.num_cross_layers,
        },
        'args': vars(args),
        'train_samples': len(train_idx),
        'val_samples': len(val_idx),
        'test_samples': len(test_dataset),
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
    }
    save_metrics_log(final_log, args.log_path)

    print("\n" + "=" * 60)
    print("训练+测试完成！")
    print("=" * 60)


if __name__ == '__main__':
    USE_HARDCODED_CONFIG = True

    if USE_HARDCODED_CONFIG:
        sys.argv = [
            "main_final.py",
            "--train_csv", "../data/CARS-AMP_train.csv",
            "--feature_dir", "../esm2_fea_CARS-AMP",
            "--test_csv", "../data/CARS-AMP_test.csv",
            "--log_path", "log/final_metrics.json",
            "--model_dir", "models_binary",
            "--use_esm",
            "--use_physico",
            "--use_peptide_gcn",
            "--use_gcn_aa_properties",
            "--pca_components", "0.95",
            "--d_model", "128",
            "--nhead", "4",
            "--num_cross_layers", "2",
            "--lr", "5e-5",
            "--batch_size", "64",
            "--epochs", "200",
            "--patience", "35",
            "--val_ratio", "0.1",
            "--max_len", "50"
        ]

    main()
