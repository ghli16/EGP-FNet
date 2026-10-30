import pandas as pd

# 输入和输出文件路径
csv_file = 'data/CARS-AMP_test.csv'
fasta_file = 'data/CARS-AMP_test.fasta'

# 读取 CSV 文件
df = pd.read_csv(csv_file)

with open(fasta_file, 'w') as f:
    for _, row in df.iterrows():
        name = row['name']
        sequence = row['sequence']
        f.write(f'>{name}\n{sequence}\n')

print(f"已将 {len(df)} 条记录写入 {fasta_file}")