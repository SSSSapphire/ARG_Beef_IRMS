import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA

# -------------------------------
# 1. 数据读取与预处理
# -------------------------------

# 读取训练数据
data = pd.read_csv('Sci01_Data_CNN_Train.csv')
X = data.iloc[:, 2:].values  # 53个变量
y = data.iloc[:, 1].values   # 分类标签
sample_names = data.iloc[:, 0].values  # 样品名称

# 对数变换（避免零值导致的错误，加入一个小常数）
X_log = np.log1p(X)  # log1p 是 log(x + 1)，用于避免零值

# 平均中心化
X_centered = X_log - np.mean(X_log, axis=0)

# 标准化变量
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 2. 建立PLS-DA模型
# -------------------------------

# PLS-DA模型
pls_da = PLSRegression(n_components=2)
pls_da.fit(X_scaled, pd.get_dummies(y))  # 使用one-hot编码的y进行拟合
X_scores = pls_da.x_scores_  # 投影后的二维得分

# -------------------------------
# 3. 二维聚类图可视化（带样品名称）
# -------------------------------

# 二维聚类图
plt.figure(figsize=(10, 6))
for class_label, color in zip(np.unique(y), ['red', 'blue']):
    mask = y == class_label
    plt.scatter(X_scores[mask, 0], X_scores[mask, 1], label=f'Class {class_label}', c=color, alpha=0.7)
for i, name in enumerate(sample_names):
    plt.text(X_scores[i, 0], X_scores[i, 1], name, fontsize=8)
plt.xlabel('PLS Component 1')
plt.ylabel('PLS Component 2')
plt.title('PLS-DA 2D Clustering')
plt.legend()
plt.grid()
plt.show()

# -------------------------------
# 4. 95%置信区间的二维散点图
# -------------------------------

def plot_confidence_ellipse(ax, data, facecolor, label):
    cov = np.cov(data.T)
    mean = np.mean(data, axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.arctan2(*eigenvectors[:, 0][::-1])
    width, height = 2 * np.sqrt(eigenvalues)
    ellipse = Ellipse(mean, width, height, angle=np.degrees(angle),
                      edgecolor='black', facecolor=facecolor, alpha=0.3, label=label)
    ax.add_patch(ellipse)

fig, ax = plt.subplots(figsize=(10, 6))
for class_label, color in zip(np.unique(y), ['red', 'blue']):
    mask = y == class_label
    ax.scatter(X_scores[mask, 0], X_scores[mask, 1], label=f'Class {class_label}', c=color, alpha=0.7)
    plot_confidence_ellipse(ax, X_scores[mask], color, f'95% CI - Class {class_label}')
plt.xlabel('PLS Component 1')
plt.ylabel('PLS Component 2')
plt.title('PLS-DA 2D with Confidence Ellipses')
plt.legend()
plt.grid()
plt.show()

# -------------------------------
# 5. 未知分类样品预测
# -------------------------------

# 读取未知分类数据
unknown_data = pd.read_csv('Sci01_Data_CNN_Test.csv')
unknown_X = unknown_data.iloc[:, 2:].values
unknown_names = unknown_data.iloc[:, 0].values

# 标准化未知样品
unknown_X_scaled = scaler.transform(unknown_X)

# 分类预测
unknown_pred = pls_da.predict(unknown_X_scaled)
# 获取分类标签的列名
class_labels = np.unique(y)  # y 是训练数据中的分类标签

# 构建未知样品预测概率的 DataFrame
unknown_class_probs = pd.DataFrame(unknown_pred, columns=class_labels)

# 获取预测的类别
unknown_class = unknown_class_probs.idxmax(axis=1)

# 打印预测结果
print("未知样品分类预测：")
for name, class_prob in zip(unknown_names, unknown_class_probs.values):
    print(f"{name}: {dict(zip(class_labels, class_prob))}")
# -------------------------------
# 6. VIP值计算与可视化
# -------------------------------

def calculate_vip(pls_model, X_scaled):
    t = pls_model.x_scores_  # 得分矩阵
    w = pls_model.x_weights_  # 权重矩阵
    q = pls_model.y_loadings_  # y 的加载矩阵

    # 计算每个成分对 X 的贡献（t**2 * q**2）
    s = np.sum(t ** 2, axis=0) * np.sum(q ** 2, axis=1)
    total_s = np.sum(s)

    # VIP计算
    vip_scores = np.sqrt(X_scaled.shape[1] * np.sum((w ** 2) * s / total_s, axis=1))  # 修改广播方向
    return vip_scores



# 计算 VIP 值
vip_scores = calculate_vip(pls_da, X_scaled)

# 将 VIP 值排序并可视化
vip_df = pd.DataFrame({'Variable': data.columns[2:], 'VIP': vip_scores})
vip_df = vip_df.sort_values(by='VIP', ascending=False)

# 绘制柱形图
plt.figure(figsize=(12, 6))
plt.bar(vip_df['Variable'], vip_df['VIP'], color='steelblue')
plt.xlabel('Variable')
plt.ylabel('VIP Score')
plt.title('VIP Scores')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# -------------------------------
# 7. PLS-DA模型的评估（Accuracy, R2, Q2）
# -------------------------------

# 1. Accuracy
# 创建 LabelEncoder 对象
label_encoder = LabelEncoder()
# 将 y（真实标签）转换为数字格式
y_encoded = label_encoder.fit_transform(y)
# PLS-DA 模型预测
y_pred = pls_da.predict(X_scaled)  # 使用模型进行预测
y_pred_labels = np.argmax(y_pred, axis=1)  # 获取预测的类别标签
# 将预测标签转换为与真实标签相同的格式（数字转为类别标签）
y_pred_labels = label_encoder.inverse_transform(y_pred_labels)

# 计算 Accuracy
accuracy = accuracy_score(y, y_pred_labels)
print(f"Accuracy: {accuracy:.4f}")

# 2. R²
r2 = r2_score(pd.get_dummies(y), y_pred)  # 使用one-hot编码后的标签和预测值计算R²
print(f"R²: {r2:.4f}")

# 3. Q²（交叉验证）
# 使用 LabelEncoder 将标签转换为数字
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
# 进行交叉验证计算 Q2 分数
q2_scores = cross_val_score(pls_da, X_scaled, y_encoded, cv=10, scoring='neg_mean_squared_error')
# 计算 Q2 分数（负均方误差的平均值）
q2 = np.mean(q2_scores)
print(f"Q2 Score: {q2:.4f}")

# -------------------------------
# 完整代码结束
# -------------------------------
