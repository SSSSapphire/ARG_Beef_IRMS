import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# 读取数据集
data = pd.read_csv('Sci01_Data_1029.csv')

# 数据预处理
X = data.iloc[:, 2:].values  # 选择53个变量列
y = data.iloc[:, 1].values  # 类别列

# 编码类别标签（CHN -> 0, ARG -> 1）
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 按8:2的比例分割数据集（训练集和测试集）
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 规范化数据
X_train = X_train.astype('float32') / np.max(X_train)
X_test = X_test.astype('float32') / np.max(X_test)

# 构建CNN模型
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(2),
    Dropout(0.2),
    Conv1D(128, 3, activation='relu'),
    MaxPooling1D(2),
    Dropout(0.2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')  # 二分类
])

# 编译模型
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 检查模型是否已存在，若存在则加载模型，否则进行训练
model_filename = "CNN_model2.h5"
try:
    model.load_weights(model_filename)
    print(f"Model '{model_filename}' loaded successfully.")
except:
    # 训练模型
    model.fit(X_train.reshape(-1, X_train.shape[1], 1), y_train, epochs=20, batch_size=32, validation_data=(X_test.reshape(-1, X_test.shape[1], 1), y_test), verbose=2)

    # 保存模型
    model.save(model_filename)
    print(f"Model '{model_filename}' saved successfully.")

# 模型评价
train_acc = model.history.history['accuracy']
test_acc = model.history.history['val_accuracy']

# 绘制训练和测试准确率图
plt.figure(figsize=(8, 6))
plt.plot(train_acc, label='Train Accuracy')
plt.plot(test_acc, label='Test Accuracy')
plt.title("Train and Test Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 模型预测
y_pred = model.predict(X_test.reshape(-1, X_test.shape[1], 1))

# 绘制混淆矩阵
cm = confusion_matrix(y_test, np.argmax(y_pred, axis=1))
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# 三维聚类图可视化
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_train)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y_train, cmap='coolwarm', label="Samples")

# 在图中显示样品名称
for i, txt in enumerate(data.iloc[:len(X_train), 0]):
    ax.text(X_pca[i, 0], X_pca[i, 1], X_pca[i, 2], txt, size=6)

plt.title("3D PCA of Training Data")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
ax.set_zlabel("PCA Component 3")
plt.legend()
plt.show()

# 分类预测（输入未知数据）
def classify_new_data(new_data):
    new_data = np.array(new_data).reshape(1, -1).astype('float32') / np.max(new_data)
    prediction = model.predict(new_data.reshape(1, -1, 1))
    class_prob = prediction[0]
    class_label = label_encoder.inverse_transform([np.argmax(class_prob)])[0]
    print(f"Predicted class: {class_label} with probabilities {class_prob}")

# 示例：假设新样品数据（53个变量）
new_sample = np.random.rand(53)  # 这里您可以替换为实际的数据
classify_new_data(new_sample)
