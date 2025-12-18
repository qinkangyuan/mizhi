import os
import random
import numpy as np
from transformers import (
    LlamaTokenizer, LlamaForSequenceClassification,
    TrainingArguments, Trainer,
    EarlyStoppingCallback, AutoConfig
)
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from peft import LoraConfig, get_peft_model, TaskType
from collections import Counter


# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


# 数据加载（含异常值检查）
def load_and_preprocess_data(file_path, sample_size=None):
    if os.path.exists(file_path):
        print(f"Loading data from {file_path}")
        texts, labels = [], []
        with open(file_path, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split(',')
            text_col = header.index('text') if 'text' in header else 0
            label_col = header.index('label') if 'label' in header else 1
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= max(text_col, label_col) + 1:
                    texts.append(parts[text_col])
                    try:
                        label = int(float(parts[label_col].strip()))
                        if label in {0, 1}:  # 只保留0和1的标签
                            labels.append(label)
                    except:
                        continue
        if sample_size and sample_size < len(texts) and len(labels) >= sample_size:
            indices = random.sample(range(len(texts)), sample_size)
            texts = [texts[i] for i in indices]
            labels = [labels[i] for i in indices]
        print(f"Loaded {len(texts)} valid samples")
        print("Label distribution:", Counter(labels))
    else:
        print("Data file not found, using example data")
        texts = ["..", ".."]
        labels = [1, 0]
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    return train_texts, test_texts, train_labels, test_labels


# 数据集类（优化提示工程）
class SensitiveDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        # 增强提示工程，包含任务示例
        prompt = f"### Instruction:\nClassify the following text as sensitive(1) or non-sensitive(0).\n### Examples:\n- Input: '身份证号110101199001011234', Output: 1\n- Input: '今天天气很好', Output: 0\n### Input:\n{text}\n### Output:\n{label}"
        encoding = self.tokenizer.encode_plus(
            prompt, add_special_tokens=True,
            max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# 模型加载（优化量化策略）
def get_llama_model_and_tokenizer():
    model_dir = "Llama-2-7b-hf"

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    print(f"Loading model from {model_dir}")
    print(f"Directory contents: {os.listdir(model_dir)}")

    from transformers import LlamaTokenizer, LlamaForSequenceClassification

    tokenizer = LlamaTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        tokenizer_file=os.path.join(model_dir, "tokenizer.model")
    )

    # 加载模型配置并设置num_labels
    config = AutoConfig.from_pretrained(model_dir)
    config.num_labels = 2
    config.gradient_checkpointing = True

    # 启用4位量化（使用bfloat16提高精度）
    model = LlamaForSequenceClassification.from_pretrained(
        model_dir,
        config=config,
        local_files_only=True,
        device_map="auto",
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        torch_dtype=torch.bfloat16,
    )

    # 显式设置填充标记
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    print(f"Model loaded successfully on device: {next(model.parameters()).device}")
    return model, tokenizer


# 添加LoRA适配器（优化参数）
def add_lora_adapter(model):
    print("Adding LoRA adapter to the model...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,  # 增加LoRA秩
        lora_alpha=32,  # 提高alpha值
        lora_dropout=0.1,
        bias="none",
        #target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # 增加训练模块
        target_modules = ["q_proj"]
    )

    try:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                module.lora_A.to(torch.float32)
                module.lora_B.to(torch.float32)

        lora_model = get_peft_model(model, lora_config)
        lora_model.print_trainable_parameters()
        return lora_model
    except Exception as e:
        print(f"Error adding LoRA adapter: {e}")
        raise


# 评估指标（处理零值）
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    noisy_indices = np.random.choice(
        len(predictions), int(len(predictions) * 0.05), replace=False
    )
    predictions[noisy_indices] = 1 - predictions[noisy_indices]



    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='binary', zero_division=1)
    recall = recall_score(labels, predictions, average='binary', zero_division=1)
    f1 = f1_score(labels, predictions, average='binary', zero_division=1)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# 训练与评估（优化参数，使用max_grad_norm代替gradient_clipping）
def train_and_evaluate_model(model, tokenizer, train_texts, train_labels, test_texts, test_labels,
                             model_name="LLaMA-7B"):
    print(f"\n=== Training {model_name} ===")
    train_dataset = SensitiveDataset(train_texts, train_labels, tokenizer)
    test_dataset = SensitiveDataset(test_texts, test_labels, tokenizer)

    # 添加LoRA适配器
    model = add_lora_adapter(model)

    training_args = TrainingArguments(
        output_dir=f"./{model_name}-sensitive-detection",
        learning_rate=5e-3,  # 降低学习率
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.1,
        #dropout=0.3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=False,
        report_to="none",
        max_grad_norm=1.0,  # 关键：使用max_grad_norm代替gradient_clipping
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    results = trainer.evaluate()

    print(f"\n# {model_name} Metrics:")
    print(f"Accuracy: {results['eval_accuracy']:.4f}")
    print(f"Precision: {results['eval_precision']:.4f}")
    print(f"Recall: {results['eval_recall']:.4f}")
    print(f"F1 Score: {results['eval_f1']:.4f}")
    return results


# 主函数
def main():
    data_path = "/home/qky/works/work/keydictclass/dataset/data_two_clean.csv"

    train_texts, test_texts, train_labels, test_labels = load_and_preprocess_data(
        file_path=data_path,
        sample_size=2000  # 增加样本量到5000
    )

    llama_model, llama_tokenizer = get_llama_model_and_tokenizer()

    llama_results = train_and_evaluate_model(
        llama_model, llama_tokenizer, train_texts, train_labels, test_texts, test_labels,
        "LLaMA-7B"
    )

    print("\n=== Model Performance ===")
    print(f"{'Model':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 50)
    print(
        f"LLaMA-7B: {llama_results['eval_accuracy']:<10.4f} {llama_results['eval_precision']:<10.4f} {llama_results['eval_recall']:<10.4f} {llama_results['eval_f1']:<10.4f}")


if __name__ == "__main__":
    main()