import torch
import esm
import os
from Bio import SeqIO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 加载 ESM-2 模型
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.to(device)
batch_converter = alphabet.get_batch_converter()
model.eval()

# 创建输出文件夹
output_folder = "esm2_fea_CARS-AMP"
os.makedirs(output_folder, exist_ok=True)
print(f"输出文件夹已准备: {output_folder}")

# 读取 FASTA 文件
fasta_file = "data/CARS-AMP_test.fasta"
data = []

# 解析 FASTA 文件
for record in SeqIO.parse(fasta_file, "fasta"):
    protein_name = record.id
    protein_sequence = str(record.seq)
    data.append((protein_name, protein_sequence))
print(f"FASTA 文件解析完成。总序列数: {len(data)}")

batch_size = 1

for batch_start in range(0, len(data), batch_size):
    print(f"\n正在处理从索引 {batch_start} 开始的批次")
    batch_data = data[batch_start: batch_start + batch_size]

    batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)

    batch_tokens = batch_tokens.to(device)

    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)

    token_representations = results["representations"][33]

    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        valid_tokens = token_representations[i, 1: tokens_len - 1]
        sequence_representations.append(valid_tokens)

    for seq_name, seq_rep in zip([x[0] for x in batch_data], sequence_representations):
        tensor_file_path = os.path.join(output_folder, f"{seq_name}.tensor")
        torch.save(seq_rep.cpu(), tensor_file_path)
        print(f"已保存 {seq_name} 的特征至 {tensor_file_path}")