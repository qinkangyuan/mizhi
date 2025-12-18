import pandas as pd

# 读取c.csv和s.csv数据集的前1万条记录
c_df = pd.read_csv('../dataset/c.csv', usecols=['label', 'text'], nrows=10000)
s_df = pd.read_csv('../dataset/s.csv', usecols=['label', 'text'], nrows=10000)

# 合并数据集
combined_df = pd.concat([c_df, s_df], ignore_index=True)

# 打乱数据集
shuffled_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 保存为data.csv
shuffled_df.to_csv('dataset/data.csv', index=False)

print("数据已合并并保存为data.csv")
