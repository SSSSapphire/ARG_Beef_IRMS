import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from matplotlib.patches import Ellipse
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

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

# Step 3: 计算VIP值
def calculate_vip(plsda, X_train):
    """
    计算每个特征的VIP值。
    plsda: 已训练好的PLS模型
    X_train: 训练数据，特征矩阵
    """
    # 获取权重矩阵 W 和特征矩阵 X
    W = plsda.x_weights_
    X_scaled = StandardScaler().fit_transform(X_train)  # 确保对数据进行标准化处理

    # 计算协方差矩阵 S
    S = np.cov(X_scaled.T)  # 协方差矩阵

    # 计算 W 的平方
    W_square = np.square(W)

    # 计算 VIP 值
    vip = np.zeros(W.shape[0])  # 创建一个空数组来存储 VIP 值
    for i in range(W.shape[0]):
        sum_weights = np.sum(W_square[i, :])  # 对每个成分的权重平方进行求和
        vip[i] = np.sqrt(np.sum(sum_weights) / np.sum(W_square))

    return vip

# Step 4: 可视化 VIP 值
def plot_vip(vip, feature_names):
    """
    绘制 VIP 值图。
    vip: VIP值数组
    feature_names: 特征名称
    """
    # 确保 vip 是一维数组
    vip = np.squeeze(vip)

    # 检查 vip 的维度
    '''
    if vip.ndim != 1:
        raise ValueError(f"VIP值的维度不正确，应该是一维数组，当前维度：{vip.ndim}")

    # 确保 vip 长度与特征名称长度一致
    if len(vip) != len(feature_names):
        raise ValueError(f"VIP值数组的长度与特征名称长度不一致: {len(vip)} != {len(feature_names)}")
    '''
    # 按 VIP 值排序
    sorted_indices = np.argsort(vip)[::-1]
    sorted_vip = vip[sorted_indices]
    sorted_feature_names = np.array(feature_names)[sorted_indices]

    # 可视化
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_feature_names, sorted_vip, color='skyblue')
    plt.xlabel('VIP Score')
    plt.ylabel('Feature')
    plt.title('VIP Scores for Each Feature')
    plt.show()




# Step 5: 二维聚类图可视化（带样品名称和95%置信区间）
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

# Step 6: 十折交叉验证
def cross_validation(X, y, plsda, scaler):
    # 使用10折交叉验证评估模型
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    accuracies = []
    r2_scores = []
    q2_scores = []

    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # 标准化数据
        X_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 训练PLS-DA模型
        plsda.fit(X_scaled, y_train)

        # 计算预测结果
        y_pred = plsda.predict(X_test_scaled)

        # 计算Accuracy
        accuracy = np.mean(y_pred == y_test)
        accuracies.append(accuracy)
        
        # 计算R2
        r2 = plsda.score(X_test_scaled, y_test)
        r2_scores.append(r2)
        
        # 计算Q2
        y_test_mean = np.mean(y_test)
        ss_total = np.sum((y_test - y_test_mean) ** 2)
        ss_residual = np.sum((y_test - y_pred) ** 2)
        q2 = 1 - ss_residual / ss_total
        q2_scores.append(q2)

    print(f"Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
    print(f"R²: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")
    print(f"Q²: {np.mean(q2_scores):.3f} ± {np.std(q2_scores):.3f}")

# Step 7: 输入未知样品并进行预测
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
    
    # 计算VIP值
    vip = calculate_vip(plsda, X_train)
    plot_vip(vip, X)
    
    # 十折交叉验证
    cross_validation(X, y, plsda, scaler)
    
    # 可视化二维聚类图
    plot_2d_clusters(X, y, names, plsda, scaler, encoder)
    
    # 输入未知样品并进行分类预测
    predict_unknown_samples(unknown_file, plsda, scaler, encoder)
