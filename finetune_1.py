import torch
from transformers import Trainer, TrainingArguments
from datapre.data_processing import load_special_tokens, initialize_tokenizer
import pandas as pd
from transformers import BertModel
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'有 {torch.cuda.device_count()} 个GPU可用.')
else:
    device = torch.device("cpu")
    print('只有CPU可用.')
# 生成输入特征
def encode_with_features(texts, special_tokens, tokenizer):
    encodings = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    # 特殊词汇计数作为特征
    special_counts = []
    for text in texts:
        tokens = tokenizer.tokenize(text)
        count = 0
        for token in tokens:
            if token in special_tokens:
                count += 1
        special_counts.append(count)
    encodings['special_counts'] = torch.tensor(special_counts).unsqueeze(1)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 检查是否有可用的GPU
    encodings = encodings.to(device)  # 将编码移至指定的设备（CPU或GPU）
    return encodings

# 构建训练数据集
class ClassifierDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, device='cuda'):
        self.encodings = encodings
        self.labels = labels
        self.device = device

    def __getitem__(self, idx):
        item = {key: val[idx].to(self.device) for key, val in self.encodings.items()}
        # 确保特殊词汇计数特征被加载并传递
        item['text'] = self.encodings['special_counts'][idx].to(self.device)
        item['labels'] = torch.tensor(self.labels[idx], device=self.device)
        return item

    def __len__(self):
        return len(self.labels)
def load_texts_from_csv(file_path):
    """从CSV文件加载文本数据"""
    df = pd.read_csv(file_path)
    texts = df['text'].tolist()  # 假设文本存储在名为'text'的列中
    return texts


class BertForSequenceClassificationWithFeatures(nn.Module):
    def __init__(self, num_labels, bert_model_name="bert_pretrain", freeze_bert=True):
        super(BertForSequenceClassificationWithFeatures, self).__init__()
        self.num_labels = num_labels
        # 加载预训练的BertModel
        self.bert = BertModel.from_pretrained(bert_model_name)

        # 冻结Bert参数，如果你不希望在训练中调整它们
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Bert的hidden_size
        self.bert_hidden_size = self.bert.config.hidden_size

        # 额外特征的维度，这里设置为1因为你有一个特殊词汇计数特征
        self.feature_dim = 1

        # 分类头
        self.classification_head = nn.Sequential(
            nn.Linear(self.bert_hidden_size + self.feature_dim, 512),  # 假设使用hidden_size加上一个特征
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_labels)
        )

    def forward(self, input_ids, attention_mask, token_type_ids, feature):
        # 获取Bert的输出
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # 取得[CLS]标记的输出用于分类
        cls_output = outputs[1]  # outputs[1] 表示获取pooled output

        # 将[CLS]标记的输出和额外特征拼接
        combined_features = torch.cat((cls_output, feature), 1)

        # 通过分类头
        logits = self.classification_head(combined_features)

        return logits

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_texts = load_texts_from_csv("dataset/data.csv")
    train_data, val_data = train_test_split(train_texts, test_size=0.2, random_state=42)

    train_labels = [0, 1]
    validation_labels = [1, 0]
    special_tokens = load_special_tokens()

    tokenizer = initialize_tokenizer(special_tokens)

    train_encodings = encode_with_features(train_data, special_tokens, tokenizer)
    validation_encodings = encode_with_features(val_data, special_tokens, tokenizer)
    train_dataset = ClassifierDataset(train_encodings, train_labels, device=device)
    validation_dataset = ClassifierDataset(validation_encodings, validation_labels, device=device)

    model = BertForSequenceClassificationWithFeatures(num_labels=2, bert_model_name="bert_pretrain",freeze_bert=True).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    print(next(model.parameters()).device)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        gradient_accumulation_steps=2,
        load_best_model_at_end=True,
        fp16=True,
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
    )

    trainer.train()

    # 预测
    predictions = trainer.predict(validation_dataset)
    preds = np.argmax(predictions.predictions, axis=1)

    # 计算评估指标
    accuracy = accuracy_score(validation_labels, preds)
    precision = precision_score(validation_labels, preds)
    recall = recall_score(validation_labels, preds)
    f1 = f1_score(validation_labels, preds)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

if __name__ == "__main__":
    main()