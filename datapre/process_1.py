import csv
import re

# 文件路径
input_file_path = '../data_B/secret-other.txt'
c_output_file_path = 'C_texts.csv'
s_output_file_path = 'S_texts.csv'

# 初始化列表来存储提取的文本
c_texts = []
s_texts = []

# 读取txt文件并处理段落

with open(input_file_path, 'r', encoding='latin1') as file:  # 使用 latin1 编码
    content = file.read()

# 分割段落
paragraphs = re.split(r'(?=C##|S##|S/NF##)', content)

# 处理每个段落
for para in paragraphs:
    para = para.strip()
    if para.startswith('C##'):
        c_texts.append(para)
    elif para.startswith('S##') or para.startswith('S/NF##'):
        s_texts.append(para)

# 将C##文本保存到CSV文件
with open(c_output_file_path, 'w', newline='', encoding='utf-8') as c_file:
    writer = csv.writer(c_file)
    for text in c_texts:
        writer.writerow([text])

# 将S##和S/NF##文本保存到CSV文件
with open(s_output_file_path, 'w', newline='', encoding='utf-8') as s_file:
    writer = csv.writer(s_file)
    for text in s_texts:
        writer.writerow([text])

print("提取和保存操作完成。")

