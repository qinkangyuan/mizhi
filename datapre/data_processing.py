import pickle
import random
from transformers import BertTokenizer, BertModel

# 加载词汇库
def load_special_tokens(s_file='extendDict/expanded_s.pkl', c_file='extendDict/expanded_c.pkl'):
    with open(s_file, 'rb') as f:
        expanded_s = pickle.load(f)
    with open(c_file, 'rb') as f:
        expanded_c = pickle.load(f)
    # 确保 expanded_s 和 expanded_c 都是集合
    if not isinstance(expanded_s, set):
        expanded_s = set(expanded_s)
    if not isinstance(expanded_c, set):
        expanded_c = set(expanded_c)

    combined_tokens = expanded_s.union(expanded_c)  # 使用 union 来合并两个集合
    return combined_tokens

# 初始化BERT分词器
def initialize_tokenizer(special_tokens):
    tokenizer = BertTokenizer.from_pretrained("bert_pretrain")
    tokenizer.add_special_tokens({'additional_special_tokens': list(special_tokens)})  # 转换为列表
    return tokenizer


# 自定义掩码策略
def tokenize_and_mask_special_tokens(text, special_tokens, tokenizer, mask_prob=0.15):
    tokens = tokenizer.tokenize(text)
    masked_tokens = []
    for token in tokens:
        if token in special_tokens and random.random() < mask_prob:
            masked_tokens.append('[MASK]')
        else:
            masked_tokens.append(token)
    return tokenizer.convert_tokens_to_ids(masked_tokens)