import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse

# 1. 数据生成
np.random.seed(42)
n_class1 = 60  # 类别1样本数
n_class2 = 40  # 类别2样本数

X_class1 = np.random.rand(n_class1, 50) + 0.5  # 类别1的数据
X_class2 = np.random.rand(n_class2, 50)        # 类别2的数据

X = np.vstack((X_class1, X_class2))            # 合并数据
y = np.hstack((np.zeros(n_class1), np.ones(n_class2)))  # 标签：0表示类1，1表示类2

# 样品名称
sample_names = [f'Sample_{i+1}' for i in range(len(y))]

# 2. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. PCA降维到2D用于可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 4. 可视化带样品名称和置信区间
def draw_confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    绘制置信椭圆，代表95%置信区间。
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    
    cov = np.cov(x, y)
    mean = np.mean(x), np.mean(y)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigvals)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, facecolor=facecolor, **kwargs)
    ax.add_patch(ellipse)

plt.figure(figsize=(12, 8))
ax = plt.gca()

# 绘制散点图
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='coolwarm', alpha=0.7, ax=ax)

# 绘制置信椭圆
draw_confidence_ellipse(X_pca[y == 0, 0], X_pca[y == 0, 1], ax, n_std=1.96, edgecolor='blue', linestyle='--', label='Class 0 - 95% CI')
draw_confidence_ellipse(X_pca[y == 1, 0], X_pca[y == 1, 1], ax, n_std=1.96, edgecolor='red', linestyle='--', label='Class 1 - 95% CI')

# 添加样品名称标签
for i, txt in enumerate(sample_names):
    ax.annotate(txt, (X_pca[i, 0], X_pca[i, 1]), fontsize=8, ha='right')

plt.title('PCA Visualization with 95% Confidence Ellipses')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='best')
plt.show()

# 5. 模型训练与预测
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)

# 预测结果和评估
y_pred = svm.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. 交叉验证
scores = cross_val_score(svm, X_scaled, y, cv=5)
print(f"Cross-validation accuracy: {scores.mean():.2f} ± {scores.std():.2f}")
