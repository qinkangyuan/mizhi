import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split
from datapre.data_processing import load_special_tokens, initialize_tokenizer, tokenize_and_mask_special_tokens
from transformers import BertModel
import pandas as pd
# 创建训练和验证数据集
def create_datasets(texts, special_tokens, tokenizer, test_size=0.1):
    train_texts, val_texts = train_test_split(texts, test_size=test_size)
    train_encodings = [tokenize_and_mask_special_tokens(text, special_tokens, tokenizer) for text in train_texts]
    val_encodings = [tokenize_and_mask_special_tokens(text, special_tokens, tokenizer) for text in val_texts]
    return train_encodings, val_encodings

# 使用数据集加载器
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return torch.tensor(self.encodings[idx])

    def __len__(self):
        return len(self.encodings)
def load_texts_from_csv(file_path):
    """从CSV文件加载文本数据"""
    df = pd.read_csv(file_path)
    texts = df['text'].tolist()  # 假设文本存储在名为'text'的列中
    return texts

def main():
    texts = load_texts_from_csv("dataset/data.csv")  # 使用新函数加载文本
    special_tokens = load_special_tokens()
    tokenizer = initialize_tokenizer(special_tokens)

    train_encodings, val_encodings = create_datasets(texts, special_tokens, tokenizer)

    train_dataset = TextDataset(train_encodings)
    val_dataset = TextDataset(val_encodings)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    # model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = BertModel.from_pretrained("bert_pretrain").to(device)

    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

if __name__ == "__main__":
    main()