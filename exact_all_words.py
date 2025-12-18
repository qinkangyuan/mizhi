import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

# 下载punkt模型，用于分词
nltk.download('punkt')

# 文件路径
input_file_path = 'train_cleaned.csv'
output_vocab_file_path = 'vocab.txt'

# 读取CSV文件
df = pd.read_csv(input_file_path)

# 初始化一个集合来存储词汇
vocab = set()

# 对每个文本字段进行分词并添加到词汇集合中
for text in df['text']:
    words = word_tokenize(text)
    vocab.update(words)

# 将词汇保存到文件，每行一个词
with open(output_vocab_file_path, 'w', encoding='utf-8') as vocab_file:
    for word in sorted(vocab):
        vocab_file.write(word + '\n')

print("词库提取完成并保存到", output_vocab_file_path)
