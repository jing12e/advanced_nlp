from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 读取数据集
df = pd.read_csv('test_dataset.csv')
# 将标签转换为整数
def label_to_integer(label_list):
    binary_string = ''.join(['1' if label != '_' else '0' for label in label_list])
    integer_value = int(binary_string, 2)
    return integer_value

# 将原始标签转换为整数标签
df['label'] = df['arguments'].apply(label_to_integer)

# 初始化 TF-IDF 向量化器
tfidf_vectorizer = TfidfVectorizer()

# 对文本数据进行 TF-IDF 向量化
X = tfidf_vectorizer.fit_transform(df['word'])

# 提取标签
y = df['label']

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化模型
model = LogisticRegression()

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 计算评估指标
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
accuracy = accuracy_score(y_test, y_pred)

# 打印评估指标
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Accuracy:", accuracy)