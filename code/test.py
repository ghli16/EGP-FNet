import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import PeptideModel
from dataset import PeptideDataset, collate_fn
from physico_features import PhysicoFeatureExtractor
import numpy as np
import json
import pandas as pd
from sklearn.metrics import (roc_auc_score, f1_score, recall_score, precision_score,
                             matthews_corrcoef, accuracy_score, average_precision_score)

import warnings

warnings.filterwarnings("ignore")

# ==================== 随机种子 ====================
seed = 42


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(seed)


# ==================== 评估指标 ====================
def compute_binary_metrics(y_true, y_pred, y_prob):
    """计算二分类评估指标"""
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()

    try:
        pr = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        nr = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        ap = average_precision_score(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
    except Exception as e:
        print(f"计算指标时出错: {e}")
        pr, nr, precision, f1, mcc, accuracy, ap, auc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    return {
        'PR': pr, 'NR': nr, 'precision': precision, 'accuracy': accuracy,
        'AP': ap, 'F1': f1, 'MCC': mcc, 'AUC': auc
    }


# ==================== 预测函数 ====================
def predict(model, data_loader, device, return_attention=False):

    model.eval()
    y_true = []
    y_prob = []
    sequences = []
    ids = []
    attention_weights = [] if return_attention else None

    with torch.no_grad():
        for batch in data_loader:
            esmFeatures, y, batch_ids, esm_lengths, physico, seq_tokens, seq_lengths = batch

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
            prob = torch.sigmoid(out).cpu().numpy()

            y_true.extend(y.cpu().numpy())
            y_prob.extend(prob)
            ids.extend(batch_ids)

            if return_attention and attn_weights is not None:
                attention_weights.extend([w.cpu().numpy() for w in attn_weights])

    y_prob = np.array(y_prob).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    predictions = {
        'ids': ids,
        'probabilities': y_prob,
        'predictions': y_pred,
    }

    if len(y_true) > 0:
        predictions['true_labels'] = np.array(y_true)
        predictions['metrics'] = compute_binary_metrics(predictions['true_labels'], y_pred, y_prob)

    if attention_weights is not None:
        predictions['attention_weights'] = attention_weights

    return predictions


# ==================== 保存预测结果 ====================
def save_predictions(predictions, output_csv):

    df = pd.DataFrame({
        'id': predictions['ids'],
        'probability': predictions['probabilities'].flatten(),
        'prediction': predictions['predictions'].flatten(),
    })

    if 'true_labels' in predictions:
        df['true_label'] = predictions['true_labels']

    df['predicted_class'] = df['prediction'].map({1: 'AMP', 0: 'Non-AMP'})

    df.to_csv(output_csv, index=False)
    print(f"预测结果已保存至: {output_csv}")

    return df


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description='测试集预测脚本')

    parser.add_argument('--model_path', required=True, help='训练好的模型路径 (.pt文件)')
    parser.add_argument('--test_csv', required=True, help='测试集CSV文件路径')
    parser.add_argument('--feature_dir', required=True, help='ESM2特征目录')

    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--threshold', type=float, default=0.5, help='分类阈值')
    parser.add_argument('--return_attention', action='store_true', default=False,
                        help='是否返回注意力权重（用于可解释性分析）')
    parser.add_argument('--max_len', type=int, default=50, help='最大序列长度')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ==================== 1. 加载模型 ====================
    print("\n=== 步骤1：加载模型 ===")
    checkpoint = torch.load(args.model_path, map_location=device)

    # 获取模型配置
    model_args = checkpoint.get('args', {})
    physico_dim = checkpoint.get('physico_dim', 0)
    pca_model_path = checkpoint.get('pca_model_path', None)
    val_metrics = checkpoint.get('val_metrics', None)

    print("模型配置信息：")
    print(f"  - 使用ESM: {model_args.get('use_esm', True)}")
    print(f"  - 使用理化特征: {model_args.get('use_physico', False)}")
    print(f"  - 使用PeptideGCN: {model_args.get('use_peptide_gcn', False)}")
    print(f"  - GCN氨基酸理化性质: {model_args.get('use_gcn_aa_properties', False)}")
    print(f"  - 融合方式: {model_args.get('fusion_type', 'cross_attn')}")
    print(f"  - d_model: {model_args.get('d_model', 128)}")
    print(f"  - 交叉注意力层数: {model_args.get('num_cross_layers', 2)}")
    print(f"  - 理化特征维度: {physico_dim}")

    if val_metrics:
        print(f"\n模型验证集性能（来自训练时）：")
        print(f"  - Accuracy: {val_metrics.get('accuracy', 'N/A')}")
        print(f"  - MCC: {val_metrics.get('MCC', 'N/A')}")

    # 创建模型
    model = PeptideModel(
        physico_dim=physico_dim,
        encoder_type=model_args.get('encoder_type', 'transformer'),
        pooling=model_args.get('pooling', 'attn'),
        fusion_type=model_args.get('fusion_type', 'cross_attn'),
        use_esm=model_args.get('use_esm', True),
        use_physico=model_args.get('use_physico', False),
        use_peptide_gcn=model_args.get('use_peptide_gcn', False),
        use_gcn_aa_properties=model_args.get('use_gcn_aa_properties', False),
        d_model=model_args.get('d_model', 128),
        nhead=model_args.get('nhead', 4),
        num_cross_layers=model_args.get('num_cross_layers', 2)
    ).to(device)

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    print("模型加载成功！")

    # ==================== 2. 加载PCA模型 ====================
    print("\n=== 步骤2：加载PCA模型 ===")
    if model_args.get('use_physico', False) and pca_model_path:
        print(f"加载PCA模型: {pca_model_path}")
        test_pca_extractor = PhysicoFeatureExtractor(pca_components=model_args.get('pca_components', 0.95))
        test_pca_extractor.load(pca_model_path)
        print(f"PCA模型加载成功，理化特征维度: {test_pca_extractor.feature_dim}")
    else:
        test_pca_extractor = None
        print("不使用理化特征或未找到PCA模型")

    # ==================== 3. 加载测试集 ====================
    print("\n=== 步骤3：加载测试集 ===")
    test_dataset = PeptideDataset(
        args.test_csv,
        args.feature_dir,
        scaler=None,
        use_esm=model_args.get('use_esm', True),
        pca_extractor=test_pca_extractor,
        max_len=args.max_len
    )
    print(f"测试集样本数: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )

    # ==================== 4. 执行预测 ====================
    print("\n=== 步骤4：执行预测 ===")
    predictions = predict(model, test_loader, device, return_attention=args.return_attention)

    if args.threshold != 0.5:
        print(f"\n使用自定义阈值: {args.threshold}")
        predictions['predictions'] = (predictions['probabilities'] > args.threshold).astype(int)
        if 'true_labels' in predictions:
            predictions['metrics'] = compute_binary_metrics(
                predictions['true_labels'],
                predictions['predictions'],
                predictions['probabilities']
            )

    # ==================== 5. 输出结果 ====================
    print("\n=== 步骤5：输出结果 ===")

    n_positive = np.sum(predictions['predictions'] == 1)
    n_negative = np.sum(predictions['predictions'] == 0)
    print(f"\n预测统计:")
    print(f"  - 阳性样本 (AMP): {n_positive} ({n_positive / len(predictions['predictions']) * 100:.1f}%)")
    print(f"  - 阴性样本 (Non-AMP): {n_negative} ({n_negative / len(predictions['predictions']) * 100:.1f}%)")

    print(f"\n预测概率分布:")
    print(f"  - 最小值: {predictions['probabilities'].min():.4f}")
    print(f"  - 最大值: {predictions['probabilities'].max():.4f}")
    print(f"  - 均值: {predictions['probabilities'].mean():.4f}")
    print(f"  - 中位数: {np.median(predictions['probabilities']):.4f}")

    if 'true_labels' in predictions:
        print(f"\n=== 模型性能（测试集） ===")
        metrics = predictions['metrics']
        print(f"  - Accuracy: {metrics['accuracy']:.4f}")
        print(f"  - Precision: {metrics['precision']:.4f}")
        print(f"  - Recall (PR): {metrics['PR']:.4f}")
        print(f"  - Specificity (NR): {metrics['NR']:.4f}")
        print(f"  - MCC: {metrics['MCC']:.4f}")
        print(f"  - F1 Score: {metrics['F1']:.4f}")
        print(f"  - AUC: {metrics['AUC']:.4f}")
        print(f"  - AP: {metrics['AP']:.4f}")
    else:
        print("\n注意: 测试集没有标签列，无法计算性能指标")

    # ==================== 6. 保存结果 ====================
    print("\n=== 步骤6：保存结果 ===")

    test_basename = os.path.splitext(os.path.basename(args.test_csv))[0]
    output_csv = f"predictions_{test_basename}.csv"

    save_predictions(predictions, output_csv)

    print("\n" + "=" * 60)
    print("预测完成！")
    print("=" * 60)



if __name__ == '__main__':
    USE_HARDCODED_CONFIG = True  

    if USE_HARDCODED_CONFIG:
        sys.argv = [
            "predict.py",
            "--model_path", "models_binary/final_model.pt",
            "--test_csv", "../data/CARS-AMP_test.csv",
            "--feature_dir", "../esm2_fea_CARS-AMP",
            "--batch_size", "64",
            "--threshold", "0.5",
            "--max_len", "50"
        ]

    main()