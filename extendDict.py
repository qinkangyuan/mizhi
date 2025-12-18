import random
import pickle
import nltk
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
"""
Basic version, updates coming soon
"""

nltk.data.path.append('nltk_data')


# Trie 树实现
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def get_all_words(self, prefix=''):
        words = []
        self._get_all_words_recursive(self.root, prefix, words)
        return words

    def _get_all_words_recursive(self, node, prefix, words):
        if node.is_end_of_word:
            words.append(prefix)
        for char, next_node in node.children.items():
            self._get_all_words_recursive(next_node, prefix + char, words)


# 替换字符
def replace_characters(word):
    replacements = set()
    for i in range(len(word)):
        replaced_word = word[:i] + '*' + word[i + 1:]
        replacements.add(replaced_word)
    return replacements


# 拆解词语
def split_word(word):
    splits = set()
    for i in range(1, len(word)):
        splits.add(word[:i] + ' ' + word[i:])
    return splits


# 生成缩写
def create_abbreviation(word):
    if len(word) <= 1:
        return {word}
    return {word[0] + "".join(random.sample(word[1:], len(word) - 1))}


# 结构化扩展
def structured_expansion(words):
    expanded_words = set(words)
    trie = Trie()

    for word in words:
        trie.insert(word)
        expanded_words.update(replace_characters(word))
        expanded_words.update(split_word(word))
        expanded_words.update(create_abbreviation(word))

    return expanded_words


# 非结构化扩展
def unstructured_expansion(words):
    expanded_words = set(words)

    similar_chars = {
        'a': ['4', '@'],
        'e': ['3'],
        'i': ['1', '!'],
        'o': ['0'],
        's': ['5', '$'],
        'g': ['9'],
    }

    def add_random_chars(word):
        positions = range(len(word) + 1)
        return {word[:i] + random.choice('abcdefghijklmnopqrstuvwxyz') + word[i:] for i in positions}

    def delete_random_chars(word):
        if len(word) == 1:
            return {word}
        positions = range(len(word))
        return {word[:i] + word[i + 1:] for i in positions}

    def replace_similar_chars(word):
        replacements = set()
        for i, char in enumerate(word):
            if char in similar_chars:
                for similar_char in similar_chars[char]:
                    replaced_word = word[:i] + similar_char + word[i + 1:]
                    replacements.add(replaced_word)
        return replacements

    for word in words:
        expanded_words.update(add_random_chars(word))
        expanded_words.update(delete_random_chars(word))
        expanded_words.update(replace_similar_chars(word))

    return expanded_words



# 加载c.pkl词典
with open('dict/c.pkl', 'rb') as f:
    c_words = pickle.load(f)



# 进行结构化扩展
structured_expanded_words = structured_expansion(c_words)

# 进行非结构化扩展
fully_expanded_words = unstructured_expansion(c_words)

# 合并扩展后的词典
final_expanded_words = structured_expanded_words.union(fully_expanded_words).union(c_words)
# print("合并后的扩展词典:", final_expanded_words)


# 去重操作
unique_dictionary = list(set(final_expanded_words))


# 保存去重后的词典
with open('extendDict/expanded_c.pkl', 'wb') as f:
    pickle.dump(unique_dictionary, f)

