import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import nltk
import pickle

nltk.download('punkt')
nltk.download('stopwords')

# 加载BERT tokenizer和模型
tokenizer = BertTokenizer.from_pretrained("bert_pretrain")
model = BertModel.from_pretrained("bert_pretrain")

# 读取CSV文件
df = pd.read_csv('s_cut_data/s3.csv')

# 提取关键字函数
def textrank_extract_keywords(text, top_k):
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    word_freq = Counter(words)
    max_freq = max(word_freq.values())
    word_freq = {word: freq / max_freq for word, freq in word_freq.items()}

    co_occurrence_matrix = {}
    window_size = 3
    for i in range(len(words)):
        for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
            if i != j:
                word_i, word_j = words[i], words[j]
                if word_i not in co_occurrence_matrix:
                    co_occurrence_matrix[word_i] = {}
                if word_j not in co_occurrence_matrix[word_i]:
                    co_occurrence_matrix[word_i][word_j] = 0
                co_occurrence_matrix[word_i][word_j] += 1

    d = 0.85
    tol = 1e-5
    scores = {word: 1.0 for word in word_freq}
    old_scores = {word: 0.0 for word in word_freq}

    while not all(abs(scores[word] - old_scores[word]) < tol for word in scores):
        old_scores = scores.copy()
        for word in scores:
            scores[word] = (1 - d) + d * sum(
                co_occurrence_matrix.get(word, {}).get(w, 0) / sum(co_occurrence_matrix.get(w, {}).values()) *
                old_scores[w] for w in co_occurrence_matrix.get(word, {}))

    top_keywords = [word for word, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]]
    return top_keywords


def calculate_euclidean_distance(e1, e2):
    return torch.norm(e1 - e2, p=2)


def extract_keywords(text, mask_token="<mask>", top_k_ratio=0.1, k_ratio=0.2):
    text_length = len(text.split())
    top_k = max(int(text_length * top_k_ratio), 1)
    k = max(int(text_length * k_ratio), 1)

    candidate_keywords = textrank_extract_keywords(text, top_k=k)

    inputs1 = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    output1 = model(**inputs1, return_dict=True)
    e1 = output1.last_hidden_state.mean(dim=1).squeeze()

    differences = []
    for candidate in candidate_keywords:
        masked_text = text.replace(candidate, mask_token)
        inputs2 = tokenizer(masked_text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        output2 = model(**inputs2, return_dict=True)
        e2 = output2.last_hidden_state.mean(dim=1).squeeze()
        distance = calculate_euclidean_distance(e1, e2)
        differences.append((candidate, distance))

    differences.sort(key=lambda x: x[1], reverse=True)
    keywords = [item[0] for item in differences[:top_k]]
    return keywords


def build_dictionary(texts, mask_token="<mask>", top_k_ratio=0.02, k_ratio=0.04):
    dictionary = set()
    for text in texts:
        keywords = extract_keywords(text, mask_token=mask_token, top_k_ratio=top_k_ratio, k_ratio=k_ratio)
        dictionary.update(keywords)
    return dictionary


def save_splitted_dictionaries(texts, batch_size=1000, output_prefix='s3_dict/s'):
    num_batches = (len(texts) + batch_size - 1) // batch_size
    for i in range(num_batches):
        batch_texts = texts[i * batch_size: (i + 1) * batch_size]
        dictionary = build_dictionary(batch_texts, top_k_ratio=0.1, k_ratio=0.2)
        output_file = f'{output_prefix}{i + 1}.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(dictionary, f)
        print(f"Saved dictionary for batch {i + 1} to {output_file}")


texts = df['text'].tolist()
save_splitted_dictionaries(texts, batch_size=1000)

# 输出前100个单词
dictionary = build_dictionary(texts[:1000], top_k_ratio=0.1, k_ratio=0.2)  # 仅处理前1000个文本以便查看前100个单词
print("Number of unique keywords:", len(dictionary))
print("First 100 words in the dictionary:", list(dictionary)[:100])
