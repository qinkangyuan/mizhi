import pandas as pd
import re

# 文件路径
input_file_path = 'C_cleaned.csv'
output_file_path = '../dataset/c.csv'

# 读取CSV文件，没有标题行
df = pd.read_csv(input_file_path, header=None, names=['raw_text'])

# 定义标签替换函数
def replace_label(text):
    if text.startswith('C##'):
        return 0, text[3:].strip()
    #elif text.startswith('SBU##'):
    #    return 1, text[5:].strip()
    #elif text.startswith('S/NF##'):
    #    return 1, text[6:].strip()
    return None, text  # 不符合条件的行

# 应用标签替换函数
df[['label', 'text']] = df['raw_text'].apply(lambda x: pd.Series(replace_label(x)))

# 删除无效行
df = df.dropna(subset=['label'])

# 保存清理后的数据到新的CSV文件
df[['label', 'text']].to_csv(output_file_path, index=False)

print("数据预处理完成并保存到", output_file_path)