import pickle
import glob

def load_pickle_files(file_pattern):
    all_keywords = set()
    files = glob.glob(file_pattern)
    for file in files:
        with open(file, 'rb') as f:
            keywords = pickle.load(f)
            all_keywords.update(keywords)
    return all_keywords

def save_keywords_to_pickle(keywords, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(keywords, f)
    print(f"Saved merged and deduplicated keywords to {output_file}")

# 指定pkl文件的模式（例如所有以s_开头的pkl文件）
file_pattern = 'c_dict/c_*.pkl'
# 读取并合并所有关键字
all_keywords = load_pickle_files(file_pattern)
# 保存去重后的关键字
save_keywords_to_pickle(all_keywords, 'dict/c_first.pkl')

# 加载词典
with open('dict/c_first.pkl', 'rb') as f:
    dictionary = pickle.load(f)

# 去重操作
unique_dictionary = list(set(dictionary))

# 保存去重后的词典
with open('../dict/c.pkl', 'wb') as f:
    pickle.dump(unique_dictionary, f)

print("Number of unique keywords:", len(all_keywords))
print("First 100 words in the dictionary:", list(all_keywords)[:100])
