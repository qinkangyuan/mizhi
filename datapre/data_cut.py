import pandas as pd

# 读取CSV文件
df = pd.read_csv('dataset/s.csv')

# 每个文件的行数
chunk_size = 10000

# 计算总共有多少个文件
num_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size != 0 else 0)

# 切分数据集并保存为多个CSV文件
for i in range(num_chunks):
    start_row = i * chunk_size
    end_row = min((i + 1) * chunk_size, len(df))
    chunk_df = df.iloc[start_row:end_row]
    output_file = f's_cut_data/s{i + 1}.csv'
    chunk_df.to_csv(output_file, index=False)
    print(f'Saved {output_file}')

print("All files are saved successfully.")
