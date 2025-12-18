import pandas as pd
import pickle
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class WeightedDictionaryClassifier:
    def __init__(self, label_1_dict, label_0_dict):
        self.dictionary = set(label_1_dict).union(set(label_0_dict))  # 合并词典
        self.tokenizer = BertTokenizer.from_pretrained('bert_pretrain')
        self.model = BertModel.from_pretrained('bert_pretrain')
        self.default_weight = 0.01  # 统一的权重

    def truncate_texts(self, text, max_length=512):
        # 将文本截断到最大长度限制内
        tokens = self.tokenizer.tokenize(text)
        truncated_tokens = tokens[:max_length]
        return self.tokenizer.convert_tokens_to_string(truncated_tokens)

    def get_embeddings(self, text):
        truncated_text = self.truncate_texts(text)  # 确保文本不会超过最大长度限制
        tokens = self.tokenizer.encode_plus(
            truncated_text,
            max_length=512,  # 确保最大长度不超过 512
            truncation=True,  # 如果超过最大长度，则截断
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.model(**tokens)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # 获取平均嵌入

    def classify(self, text):
        words = text.split()
        weights = []

        # 修正权重分配
        for word in words:
            if word in self.dictionary:
                weights.append(self.default_weight)  # c.pkl 比重更高


        embedding = self.get_embeddings(text)  # BERT嵌入

        # 取权重和embedding最小长度，以防数组长度不匹配
        min_len = min(len(weights), len(embedding))
        weights = np.array(weights[:min_len])
        embedding = embedding[:min_len]

        total_weight = weights.sum() if len(weights) > 0 else 1  # 防止除零错误

        score = np.dot(embedding, weights) / total_weight if total_weight > 0 else 0  # 归一化得分

        return 0.0 if score < 1.0 else 1.0  # 根据阈值进行分类


def load_word_list(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)


# 读取词典
label_1_dict = load_word_list("dict/s.pkl")  # 秘密级别词典
label_0_dict = load_word_list("dict/c.pkl")  # 机密级别词典

# 初始化分类器
classifier = WeightedDictionaryClassifier(label_1_dict, label_0_dict)

# 读取数据集
data = pd.read_csv("dataset/data.csv")

# 进行预测
data['predicted_label'] = data['text'].apply(classifier.classify)

# 计算评估指标
accuracy = accuracy_score(data['label'], data['predicted_label'])
precision = precision_score(data['label'], data['predicted_label'], zero_division=1)
recall = recall_score(data['label'], data['predicted_label'], zero_division=1)
f1 = f1_score(data['label'], data['predicted_label'], zero_division=1)

# 输出评估指标
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# 输出结果
output_file = "result/classified_data.csv"
data.to_csv(output_file, index=False)
print(f"分类结果已保存到 {output_file}")