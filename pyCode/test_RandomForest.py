import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from matplotlib.patches import Ellipse

# Step 1: 读取数据并准备训练集
def load_data(csv_file):
    data = pd.read_csv(csv_file)
    X = data.iloc[:, 2:].values  # 特征列，假设前两列是Name和Class
    y = data.iloc[:, 1].values   # 类别标签列
    names = data.iloc[:, 0].values  # 样品名称列
    return X, y, names

# Step 2: 随机森林模型训练
def train_random_forest(X_train, y_train):
    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # 将类别标签编码
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_train)
    
    # 创建并训练随机森林分类器
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y_encoded)

    return rf, scaler, encoder

# Step 3: 二维聚类图可视化（带样品名称和95%置信区间）
def plot_2d_clusters(X, y, names, rf, scaler, encoder):
    # 标准化数据
    X_scaled = scaler.transform(X)

    # 使用随机森林模型进行预测
    X_rf = rf.apply(X_scaled)  # 获取每个样本在树上的路径

    # 使用 LabelEncoder 转换类别标签为数值型
    y_encoded = encoder.transform(y)

    # 绘制散点图，使用类别数值作为颜色映射
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_rf[:, 0], X_rf[:, 1], c=y_encoded, cmap='coolwarm', edgecolor='k', s=100)

    # 标记样品名称
    for i, name in enumerate(names):
        plt.text(X_rf[i, 0], X_rf[i, 1], name, fontsize=9)

    # 计算95%置信区间并绘制
    mean = np.mean(X_rf, axis=0)
    
    # 为每个类别绘制不同颜色的置信区间
    unique_classes = np.unique(y_encoded)
    colors = ['blue', 'orange']  # 可以根据需要调整颜色

    for i, cls in enumerate(unique_classes):
        # 获取该类别的样本
        class_samples = X_rf[y_encoded == cls]

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
        cls = int(cls)  # 强制将 cls 转换为整数类型，确保它是一个有效的索引
        plt.scatter(class_samples[:, 0], class_samples[:, 1], label=f"Class {encoder.classes_[cls]}", 
                    color=colors[i], edgecolor='black', s=100)

    plt.title("Random Forest 2D Cluster Plot with 95% Confidence Ellipse")
    plt.xlabel('Random Forest Feature 1')
    plt.ylabel('Random Forest Feature 2')
    plt.legend()
    plt.show()

# Step 4: 输入未知样品并进行预测
def predict_unknown_samples(unknown_csv, rf, scaler, encoder):
    unknown_data = pd.read_csv(unknown_csv)
    X_unknown = unknown_data.iloc[:, 2:].values  # 特征列，假设前两列是Name和Class
    names_unknown = unknown_data.iloc[:, 0].values  # 样品名称列
    
    # 标准化未知样品数据
    X_unknown_scaled = scaler.transform(X_unknown)
    
    # 进行预测
    predicted_classes = rf.predict(X_unknown_scaled)
    
    # 计算每个样品的类别概率
    probabilities = rf.predict_proba(X_unknown_scaled)
    
    print("Predicted Probabilities for Unknown Samples:")
    for i, name in enumerate(names_unknown):
        print(f"Sample {name}:")
        for j, label in enumerate(encoder.classes_):
            print(f"  Class: {label}, Probability: {probabilities[i, j]:.2f}")
        print()

# 示例执行
if __name__ == "__main__":
    # 训练集的CSV文件路径
    training_file = 'C:\\Users\\amd9600\\Desktop\\beef_sci01\\origin_data_country_modify.csv'  # 这里修改为你自己的训练数据路径
    unknown_file = 'C:\\Users\\amd9600\\Desktop\\beef_sci01\\unknown_samples.csv'  # 这里修改为你自己的未知样品数据路径

    # 加载训练数据
    X, y, names = load_data(training_file)
    
    # 划分训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练 随机森林模型
    rf, scaler, encoder = train_random_forest(X_train, y_train)
    
    # 可视化二维聚类图
    plot_2d_clusters(X, y, names, rf, scaler, encoder)
    
    # 输入未知样品并进行分类预测
    predict_unknown_samples(unknown_file, rf, scaler, encoder)
