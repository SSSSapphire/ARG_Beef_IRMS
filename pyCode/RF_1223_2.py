import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# 1. 加载训练数据
data = pd.read_csv('Sci01_Data_CNN_Train.csv')

# 提取特征和标签
X = data.iloc[:, 2:]  # 从第3列开始为53个变量
y = data['Location']  # 类别标签为 'Location'

# 使用 LabelEncoder 将类别标签编码为数字
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 划分训练集和测试集（80%训练，20%测试）
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 2. 构建随机森林分类模型
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 保存模型（以pkl格式保存，h5格式不适用于RandomForest）
model_dir = 'RF_model'
os.makedirs(model_dir, exist_ok=True)
joblib.dump(rf_model, os.path.join(model_dir, 'RF_model.pkl'))

# 输出训练准确率
train_accuracy = rf_model.score(X_train, y_train)
print(f'Training accuracy: {train_accuracy:.4f}')

# 3. 模型评价：预测测试集并计算准确率
y_pred = rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f'Test accuracy: {test_accuracy:.4f}')

# 绘制训练和测试准确率的图表
accuracies = [train_accuracy, test_accuracy]
accuracy_labels = ['Training Accuracy', 'Test Accuracy']
plt.bar(accuracy_labels, accuracies, color=['blue', 'orange'])
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.show()

# 随机森林模型特征重要性可视化
feature_importances = rf_model.feature_importances_
features = X.columns

# 按重要性排序
sorted_idx = np.argsort(feature_importances)[::-1]
plt.figure(figsize=(10, 6))
plt.barh(features[sorted_idx], feature_importances[sorted_idx], color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importances')
plt.show()

# 4. 二维PCA聚类图
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

# 可视化训练集数据
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_train, palette='Set2', style=y_train, s=100)
'''for i, name in enumerate(data['Name']):
    plt.text(X_pca[i, 0], X_pca[i, 1], name, fontsize=8)'''
plt.title('PCA of Training Data')
plt.show()

# 5. 分类预测：对未知样品进行分类预测
# 假设有一个名为 unknown_samples.csv 的未知数据集，其中没有 'Location' 列
unknown_data = pd.read_csv('Sci01_Data_CNN_Test.csv')

# 选择未知数据中的特征列（不包括 'Location' 列）
X_unknown = unknown_data.drop(columns=['Name', 'Location'], errors='ignore')  # 删除非特征列

# 确保未知数据的特征列与训练数据相同（顺序、数量一致）
X_unknown = X_unknown[X.columns]  # 保证列的顺序和数量一致

# 预测未知样品的类别
predictions = rf_model.predict(X_unknown)

# 输出预测的概率
prediction_probabilities = rf_model.predict_proba(X_unknown)
prediction_percentages = pd.DataFrame(prediction_probabilities, columns=label_encoder.classes_)

# 将预测结果添加到未知样品数据中
unknown_data['Predicted Location'] = label_encoder.inverse_transform(predictions)
unknown_data['Predicted Percentages'] = prediction_percentages.max(axis=1)

# 保存结果为 CSV 文件
unknown_data.to_csv('RF_predicted_unknown_samples.csv', index=False)

print("Prediction completed. Results saved to 'RF_predicted_unknown_samples.csv'.")
