import pandas as pd
import numpy as np

# 读取两个CSV文件
file1 = pd.read_csv("dataset/c.csv")
file2 = pd.read_csv("dataset/s.csv")

# 选择前68000条数据
data1 = file1.head(68000)
data2 = file2.head(68000)

# 合并两个数据集
combined_data = pd.concat([data1, data2], ignore_index=True)

# 打乱数据
shuffled_data = combined_data.sample(frac=1).reset_index(drop=True)

# 保存到新的CSV文件
shuffled_data.to_csv("dataset/bigdata.csv", index=False)

print("合并和打乱完成")