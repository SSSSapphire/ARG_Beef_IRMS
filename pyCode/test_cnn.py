import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


# 1. 数据准备
num_samples = 20
num_features = 10
num_classes = 2

# 读取csv
data =np.genfromtxt('Sci01_Data_1029.csv',delimiter=',',dtype=None,encoding='utf-8')

# 读取标签和名称
sample_names = data[1:,0].tolist()
sample_labels = data[1:,1]

#label转换为数值并进行独热编码
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(sample_labels)
labels_catagorical = to_categorical(labels_encoded)

# 修建与重塑数据为适合CNN的输入格式
data = data[1:,2:]
data = data.astype(np.float32)
data_reshaped = data.reshape(83,53,1,1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data_reshaped, labels_catagorical, test_size=0.2, random_state=42)

# 构建 CNN 模型以提取特征
model = Sequential()
model.add(Conv2D(32, kernel_size=(2, 1), activation='relu', input_shape=(data_reshaped.shape[1], 1, 1)))
model.add(MaxPooling2D(pool_size=(1, 1)))  # 可以根据需要调整
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))  # 输出层

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(X_train, y_train, epochs=500, batch_size=8, verbose=1, validation_data=(X_test, y_test))

# 评估模型
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test accuracy: {test_accuracy:.4f}')

# 预测
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# 打印分类报告
print(classification_report(true_classes, predicted_classes, target_names=label_encoder.classes_))

# 可视化训练过程
plt.figure(figsize=(12, 5))


# 绘制准确率
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# 绘制损失
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()

conf_matrix = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()

pca = PCA(n_components=2)
data_reduced = pca.fit_transform(data)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=data_reduced[:, 0], y=data_reduced[:, 1], hue=sample_labels, style=sample_labels, palette='Set1')
plt.title('2D Distribution of Samples')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Class')
plt.grid()
plt.show()