import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from collections import Counter
import nltk

# 设置nltk数据路径
nltk.data.path.append('nltk_data')

# 读取数据集
df = pd.read_csv('dataset/data.csv')

# 加载词典
with open('extendDict/expanded_s.pkl', 'rb') as f:
    dict_label_1 = pickle.load(f)

with open('extendDict/expanded_c.pkl', 'rb') as f:
    dict_label_0 = pickle.load(f)

# 特征提取函数
def extract_features(texts, dict_label_1, dict_label_0):
    features = []
    for text in texts:
        words = text.split()
        label_1_count = sum(1 for word in words if word in dict_label_1)
        label_0_count = sum(1 for word in words if word in dict_label_0)
        features.append([label_1_count, label_0_count])
    return pd.DataFrame(features, columns=['label_1_count', 'label_0_count'])

# 提取特征
texts = df['text'].tolist()
labels = df['label'].tolist()
features = extract_features(texts, dict_label_1, dict_label_0)

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 构建并训练模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 预测并评估
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
