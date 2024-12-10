import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from matplotlib.patches import Ellipse
from scipy.stats import t

# 1. 加载训练数据
train_data = pd.read_csv('C:\\Users\\amd9600\\Desktop\\beef_sci01\\origin_data_country_modify.csv')  # 替换为训练集路径

# 提取训练集样品名称、类别和变量
train_sample_names = train_data.iloc[:, 0].values  # 第一列为样品名称
train_y = train_data.iloc[:, 1].values            # 第二列为类别
train_X = train_data.iloc[:, 2:].values           # 第三列到最后一列为变量

# 数据标准化
scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)

# 将类别转换为数字编码
encoder = LabelEncoder()
train_y_encoded = encoder.fit_transform(train_y)

# 2. 建立PLS-DA模型
pls = PLSRegression(n_components=2)  # 使用两个主成分
pls.fit(train_X_scaled, train_y_encoded)

# 获取PLS得分
X_pls = pls.transform(train_X_scaled)  # 得到前两主成分的得分

# 创建PLS得分的DataFrame
pls_scores = pd.DataFrame(X_pls, columns=['PC1', 'PC2'])
pls_scores['Label'] = train_y
pls_scores['SampleName'] = train_sample_names

# 3. 绘制二维聚类图并标记样品名称
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=pls_scores['PC1'], 
    y=pls_scores['PC2'], 
    hue=pls_scores['Label'], 
    palette='Set1', 
    s=100
)

# 标注样品名称
for i, txt in enumerate(pls_scores['SampleName']):
    plt.text(pls_scores['PC1'][i], pls_scores['PC2'][i], txt, fontsize=9, alpha=0.8)

# 计算95%置信区间
confidence_interval = 0.95
t_value = t.ppf(confidence_interval, len(train_X) - 1)

# 定义放大因子
amplification_factor = 0.9  # 放大置信区间范围

# 分组计算置信区间
grouped = pls_scores.groupby('Label')
for label, group in grouped:
    mean_pc1 = group['PC1'].mean()
    mean_pc2 = group['PC2'].mean()
    std_pc1 = group['PC1'].std()
    std_pc2 = group['PC2'].std()
    
    # 放大置信区间
    error_pc1 = amplification_factor * t_value * std_pc1
    error_pc2 = amplification_factor * t_value * std_pc2
    
    # 绘制椭圆形置信区间
    ellipse = Ellipse(
        (mean_pc1, mean_pc2),  # 中心点
        width=2 * error_pc1,   # 横轴直径
        height=2 * error_pc2,  # 纵轴直径
        alpha=0.2, 
        color=sns.color_palette('Set1')[encoder.transform([label])[0]]
    )
    plt.gca().add_artist(ellipse)

# 图表设置
plt.title('PLS-DA 2D Clustering with 95% Confidence Interval', fontsize=16)
plt.xlabel('PC1', fontsize=14)
plt.ylabel('PC2', fontsize=14)
plt.legend(title='Class')
plt.grid(True)
plt.show()

# 4. 加载未知样品数据
unknown_data = pd.read_csv('C:\\Users\\amd9600\\Desktop\\beef_sci01\\unknown_samples.csv')  # 替换为未知样品路径

# 提取未知样品名称和变量（忽略类别列）
unknown_sample_names = unknown_data.iloc[:, 0].values  # 第一列为样品名称
unknown_X = unknown_data.iloc[:, 2:].values           # 第三列到最后一列为变量

# 检查变量数是否一致
if unknown_X.shape[1] != train_X.shape[1]:
    raise ValueError(f"未知样品变量数 ({unknown_X.shape[1]}) 与模型变量数 ({train_X.shape[1]}) 不一致！")

# 对未知样品数据进行标准化
unknown_X_scaled = scaler.transform(unknown_X)

# 确保未知样品是二维数组
if unknown_X_scaled.ndim == 1:
    unknown_X_scaled = unknown_X_scaled.reshape(1, -1)

# 使用PLS模型进行预测
predicted_class_scores = pls.predict(unknown_X_scaled)

# 确保预测结果为二维数组
if predicted_class_scores.ndim == 1:
    predicted_class_scores = predicted_class_scores.reshape(1, -1)

# 计算分类概率
probabilities = predicted_class_scores / predicted_class_scores.sum(axis=1, keepdims=True)

# 输出预测分类概率
print("Predicted Class Probabilities for the Unknown Samples:")
for i, sample_prob in enumerate(probabilities):
    print(f"Sample {unknown_sample_names[i]}:")
    
    # 如果sample_prob是一维数组（即只有一个样品）
    if sample_prob.ndim == 1:
        # 确保打印所有类别的概率
        for j, label in enumerate(encoder.classes_):
            # 如果只有一个类别，可能会导致索引越界，进行检查
            if j < len(sample_prob):  # 先检查索引是否越界
                print(f"  Class: {label}, Probability: {sample_prob[j]:.2f}")
            else:
                print(f"  Class: {label}, Probability: 0.00")
    else:
        # 如果有多个样品，输出每个样品的概率
        for j, label in enumerate(encoder.classes_):
            print(f"  Class: {label}, Probability: {sample_prob[0][j]:.2f}")
