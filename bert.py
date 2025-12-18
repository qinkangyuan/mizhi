import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class BertClassifier:
    def __init__(self, model_name='bert_pretrain', num_labels=2):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess_text(self, text, max_length=512):
        return self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

    def classify(self, text):
        inputs = self.preprocess_text(text)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()
        return predicted_label

# 读取数据集
data = pd.read_csv("dataset/data.csv")  # 假设数据集中有 'text' 和 'label' 列

# 初始化分类器
classifier = BertClassifier()

# 进行预测
data['predicted_label'] = data['text'].apply(classifier.classify)

# 计算评估指标
accuracy = accuracy_score(data['label'], data['predicted_label'])
precision = precision_score(data['label'], data['predicted_label'], average='binary', zero_division=1)
recall = recall_score(data['label'], data['predicted_label'], average='binary', zero_division=1)
f1 = f1_score(data['label'], data['predicted_label'], average='binary', zero_division=1)

# 输出评估指标
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# 输出结果
output_file = "result/classified_data.csv"
data.to_csv(output_file, index=False)
print(f"分类结果已保存到 {output_file}")