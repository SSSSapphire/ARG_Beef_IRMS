import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import cross_val_score
from matplotlib.patches import Ellipse

# Step 1: 读取数据并准备训练集
def load_data(csv_file):
    data = pd.read_csv(csv_file)
    X = data.iloc[:, 2:].values  # 特征列，假设前两列是Name和Class
    y = data.iloc[:, 1].values   # 类别标签列
    names = data.iloc[:, 0].values  # 样品名称列
    return X, y, names

# Step 2: PLS-DA 模型训练
def train_plsda(X_train, y_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)  # 标准化数据

    # 将类别标签编码
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_train)

    plsda = PLSRegression(n_components=2)
    plsda.fit(X_scaled, y_encoded)

    return plsda, scaler, encoder

# Step 3: 二维聚类图可视化（带样品名称和95%置信区间）
def plot_2d_clusters(X, y, names, plsda, scaler, encoder):
    # 标准化数据
    X_scaled = scaler.transform(X)

    # 使用 PLS 模型进行预测
    X_pls = plsda.transform(X_scaled)
    
    # 使用 LabelEncoder 转换类别标签为数值型
    y_encoded = encoder.transform(y)

    # 绘制散点图，使用类别数值作为颜色映射
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_pls[:, 0], X_pls[:, 1], c=y_encoded, cmap='coolwarm', edgecolor='k', s=100)

    # 计算95%置信区间并绘制
    mean = np.mean(X_pls, axis=0)
    
    # 为每个类别绘制不同颜色的置信区间
    unique_classes = np.unique(y_encoded)
    colors = ['blue', 'orange']  # 可以根据需要调整颜色

    for i, cls in enumerate(unique_classes):
        # 获取该类别的样本
        class_samples = X_pls[y_encoded == cls]

        # 计算该类别的均值和协方差矩阵
        class_mean = np.mean(class_samples, axis=0)
        class_cov = np.cov(class_samples.T)

        eigvals, eigvecs = np.linalg.eigh(class_cov)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        amplifierNum = 1.45
        v = amplifierNum * np.sqrt(5.991 * eigvals)  # 95%置信区间，卡方分布的临界值
        u = eigvecs

        # 绘制椭圆
        ellipse = Ellipse(class_mean, v[0], v[1], angle=np.degrees(np.arctan2(u[1, 0], u[0, 0])), 
                          edgecolor=colors[i], facecolor='none', linestyle='--', linewidth=2)
        plt.gca().add_patch(ellipse)

        # 绘制该类别的样品点
        plt.scatter(class_samples[:, 0], class_samples[:, 1], label=f"Class {encoder.classes_[cls]}", 
                    color=colors[i], edgecolor='black', s=100)

    plt.title("PLS-DA 2D Cluster Plot with 95% Confidence Ellipse")
    plt.xlabel('PLS1')
    plt.ylabel('PLS2')
    plt.legend()
    plt.show()

# Step 4: 输入未知样品并进行预测
def predict_unknown_samples(unknown_csv, plsda, scaler, encoder):
    unknown_data = pd.read_csv(unknown_csv)
    X_unknown = unknown_data.iloc[:, 2:].values  # 特征列，假设前两列是Name和Class
    names_unknown = unknown_data.iloc[:, 0].values  # 样品名称列
    
    # 标准化未知样品数据
    X_unknown_scaled = scaler.transform(X_unknown)
    
    # 进行预测
    predicted_scores = plsda.predict(X_unknown_scaled)
    
    # 如果是单一列的预测结果（例如：二分类时可能只有一列得分），则扩展为二维
    if predicted_scores.ndim == 1:  
        predicted_scores = predicted_scores[:, np.newaxis]
    
    # 如果是二分类问题，确保预测结果是二维数组
    if predicted_scores.shape[1] == 1:  
        # 扩展为二分类形式，假设只有一个得分列，计算另一类的得分
        predicted_scores = np.hstack([1 - predicted_scores, predicted_scores])
    
    # 计算每个样品的类别概率
    probabilities = np.exp(predicted_scores) / np.sum(np.exp(predicted_scores), axis=1, keepdims=True)
    
    print("Predicted Probabilities for Unknown Samples:")
    for i, name in enumerate(names_unknown):
        print(f"Sample {name}:")
        for j, label in enumerate(encoder.classes_):
            print(f"  Class: {label}, Probability: {probabilities[i, j]:.2f}")
        print()

# Step 5: 计算VIP值
def calculate_vip(plsda, X_train):
    X_scaled = StandardScaler().fit_transform(X_train)  # 标准化数据
    T = plsda.x_scores_  # 得分矩阵
    W = plsda.x_weights_  # 权重矩阵
    Q = plsda.y_loadings_  # y加载矩阵
    
    # 计算VIP值
    VIP = np.sqrt(X_scaled.shape[1] * np.sum((W ** 2) * np.sum(T ** 2, axis=0) / np.sum(T ** 2), axis=1))
    
    return VIP

# Step 6: PLS-DA 模型评估
def evaluate_plsda(plsda, X_train, y_train):
    # 将 y_train 转换为数值标签
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_train)

    # PLS-DA 模型预测（回归得分）
    y_pred = plsda.predict(X_train)  # 使用模型进行预测

    # 对于二分类任务，假设 y_pred[:, 0] 是类别1的概率，y_pred[:, 1] 是类别2的概率
    # 你可以将回归得分转化为类别
    if y_pred.ndim == 1:  # 如果是单维（回归任务）
        y_pred_labels = (y_pred > 0).astype(int)  # 二分类：使用 0 作为阈值
    else:  # 如果是多维（分类任务）
        y_pred_labels = np.argmax(y_pred, axis=1)  # 获取预测的类别标签

    # 计算 Accuracy
    accuracy = accuracy_score(y_encoded, y_pred_labels)
    print(f"Accuracy: {accuracy:.4f}")

    # 计算 R²
    r2 = r2_score(y_encoded, y_pred_labels)  # 计算 R²
    print(f"R²: {r2:.4f}")

    # 计算 Q²（交叉验证）
    q2_scores = cross_val_score(plsda, X_train, y_encoded, cv=10, scoring='neg_mean_squared_error')
    q2 = np.mean(q2_scores)
    print(f"Q2 Score: {q2:.4f}")

# Step 7: 可视化VIP值
def plot_vip(VIP, feature_names):
    vip_df = pd.DataFrame({'Feature': feature_names, 'VIP': VIP})
    vip_df = vip_df.sort_values(by='VIP', ascending=False)
    
    # 绘制柱形图
    plt.figure(figsize=(12, 6))
    plt.bar(vip_df['Feature'], vip_df['VIP'], color='steelblue')
    plt.xlabel('Feature')
    plt.ylabel('VIP Score')
    plt.title('VIP Scores of Features')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# 示例执行
if __name__ == "__main__":
    # 训练集的CSV文件路径
    training_file = "Sci01_Data_CNN_Train.csv"  # 这里修改为你自己的训练数据路径
    unknown_file = "Sci01_Data_CNN_Test.csv"  # 这里修改为你自己的未知样品数据路径

    # 加载训练数据
    X, y, names = load_data(training_file)
    
    # 划分训练集与测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练 PLS-DA 模型
    plsda, scaler, encoder = train_plsda(X_train, y_train)
    
    # 可视化二维聚类图
    plot_2d_clusters(X, y, names, plsda, scaler, encoder)
    
    # 计算并显示VIP值
    VIP = calculate_vip(plsda, X_train)
    
    # 假设特征名称是特征列的索引（可以根据你的数据调整）
    feature_names = [f"Feature {i+1}" for i in range(X_train.shape[1])]
    
    # 可视化VIP值
    plot_vip(VIP, feature_names)
    
    # PLS-DA 模型评估
    evaluate_plsda(plsda, X_train, y_train)
    
    # 输入未知样品并进行分类预测
    predict_unknown_samples(unknown_file, plsda, scaler, encoder)