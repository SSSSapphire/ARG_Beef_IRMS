import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# 文件路径
train_file = "Sci01_Data_CNN_Train.csv"
test_file = "Sci01_Data_CNN_Test.csv"

# 1. 读取训练数据
train_data = pd.read_csv(train_file)
X_train = train_data.iloc[:, 2:].values  # 第三列开始为变量
y_train = train_data.iloc[:, 1].values  # 第二列为类别

# 读取测试数据
test_data = pd.read_csv(test_file)
X_test = test_data.iloc[:, 2:].values

# 2. 模型构建
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 保存模型
joblib.dump(rf_model, "RF_model.h5")

# 3. 模型评价
# 训练集和测试集的准确率
train_accuracy = rf_model.score(X_train, y_train)
print(f"Training Accuracy: {train_accuracy:.2f}")

if '类别' in test_data.columns:
    y_test = test_data.iloc[:, 1].values
    test_accuracy = rf_model.score(X_test, y_test)
    print(f"Testing Accuracy: {test_accuracy:.2f}")

    # 输出分类报告
    y_pred = rf_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # 混淆矩阵可视化
    cm = confusion_matrix(y_test, y_pred)
    def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    plot_confusion_matrix(cm, classes=np.unique(y_train), title='Confusion Matrix')

# 特征重要性可视化
importance = rf_model.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(range(len(importance)), importance, align='center')
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()

# 4. 分类图
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=y_train, palette=['blue', 'orange'], s=100)
for i, name in enumerate(train_data.iloc[:, 0]):
    plt.text(X_train_pca[i, 0], X_train_pca[i, 1], name, fontsize=9)
plt.title("2D Clustering with Original Labels")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Category", labels=["CHN", "ARG"])
plt.show()

# 5. 分类预测
def predict_new_samples(test_file, model_file="RF_model.h5"):
    new_data = pd.read_csv(test_file)
    X_new = new_data.iloc[:, 2:].values

    model = joblib.load(model_file)

    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new)

    new_data['Predicted Label'] = predictions
    prob_df = pd.DataFrame(probabilities, columns=[f"Prob_{cls}" for cls in model.classes_])
    new_data = pd.concat([new_data, prob_df], axis=1)

    output_file = "RF_Prediction_Results.csv"
    new_data.to_csv(output_file, index=False)
    print(f"Prediction results saved to {output_file}")

# 示例调用分类预测
predict_new_samples(test_file)
