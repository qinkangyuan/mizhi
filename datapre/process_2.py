import csv

# 文件路径
input_file_path = 'S_texts.csv'
c_output_file_path = 'data/S.csv'

# 初始化列表来存储提取的文本
c_texts = []

# 读取txt文件
with open(input_file_path, 'r', encoding='latin1') as file:  # 使用 latin1 编码
    lines = file.readlines()

# 处理每一行
for line in lines:
    if line.startswith('S##') or line.startswith('"S##') or line.startswith('"S/NF##') or line.startswith('SBU##') or line.startswith('S/NF##'):
    #if line.startswith('C##') or line.startswith('"C##'):
        c_texts.append(line.strip())

# 将C##文本保存到CSV文件
with open(c_output_file_path, 'w', newline='', encoding='utf-8') as c_file:
    writer = csv.writer(c_file)
    for text in c_texts:
        writer.writerow([text])

print("提取和保存操作完成。")
