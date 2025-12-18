import pandas as pd
import re

# 文件路径
input_file_path = 'data/S.csv'
output_file_path = 'S_cleaned.csv'

# 读取CSV文件，没有标题行
df = pd.read_csv(input_file_path, header=None, names=['text'])




def clean_text(text):
    if isinstance(text, str):
        # 去除特殊符号，只保留字母、数字和常用标点符号
        # text = re.sub(r'[^A-Za-z0-9,.?!;:()\'\"\s]', '', text)
        text = re.sub(r'[-"><?()]', '', text)
        # 去除多余的空格
        text = re.sub(r'\s+', ' ', text).strip()
    return text

# 对每个文本字段进行清理
# 假设文本数据在 'text' 列中，根据你的文件实际列名修改
df['text'] = df['text'].apply(clean_text)


# 保存清理后的数据到新的CSV文件
df.to_csv(output_file_path, index=False, header=False)

print("数据预处理完成并保存到", output_file_path)
