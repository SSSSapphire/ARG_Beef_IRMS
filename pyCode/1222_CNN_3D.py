import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.manifold import TSNE

# 数据预处理函数
def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    names = data.iloc[:, 0]  # 样品名称
    labels = data.iloc[:, 1]  # 类别标签
    features = data.iloc[:, 2:].values  # 特征
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)  # 将类别转为数值
    return names, features, labels_encoded, le, labels

# 构建CNN模型
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 1. 生成CNN模型评价图
def plot_evaluation(y_true, y_pred, history):
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred.argmax(axis=1))
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_pred[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    # 训练过程曲线
    plt.figure()
    plt.plot(history.history['accuracy'], label="Train Accuracy")
    plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.legend()
    plt.show()

# 2. 三维聚类可视化
def visualize_clusters(features, labels, names):
    tsne = TSNE(n_components=3, random_state=42)
    reduced_features = tsne.fit_transform(features)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 聚类可视化
    colors = ['blue' if lbl == 0 else 'red' for lbl in labels]
    scatter = ax.scatter(reduced_features[:, 0], reduced_features[:, 1], reduced_features[:, 2], c=colors, alpha=0.6)

    # 样品名称标注
    for i, name in enumerate(names):
        ax.text(reduced_features[i, 0], reduced_features[i, 1], reduced_features[i, 2], name, fontsize=8)

    # 置信区间（基于点密度）
    density = gaussian_kde(reduced_features.T)
    density_vals = density(reduced_features.T)
    scatter = ax.scatter(reduced_features[:, 0], reduced_features[:, 1], reduced_features[:, 2],
                          c=density_vals, cmap='coolwarm', alpha=0.6)

    ax.set_title("3D Clustering with Confidence Intervals")
    plt.colorbar(scatter, label="Density")
    plt.show()

# 3. 未知样品分类
def classify_unknown_samples(model, scaler, file_path, le):
    names, features, _ = preprocess_data(file_path)
    features = scaler.transform(features)
    predictions = model.predict(features)
    predicted_classes = predictions.argmax(axis=1)
    decoded_classes = le.inverse_transform(predicted_classes)

    results = pd.DataFrame({'Name': names, 'Predicted Class': decoded_classes, 'Confidence': predictions.max(axis=1)})
    print(results)
    return results

# 主程序执行
if __name__ == "__main__":
    # 文件路径
    training_file = "Sci01_Data_1029.csv"
    unknown_file = "unknown_samples.csv"

    # 数据处理
    names, features, labels, le ,labels_name = preprocess_data(training_file)

    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

    # 独热编码标签
    y_train_onehot = to_categorical(y_train)
    y_test_onehot = to_categorical(y_test)

    # CNN模型训练
    input_shape = (features_scaled.shape[1], 1)
    model = create_cnn_model(input_shape, num_classes=len(np.unique(labels)))
    X_train_reshaped = X_train.reshape(-1, input_shape[0], 1)
    X_test_reshaped = X_test.reshape(-1, input_shape[0], 1)
    history = model.fit(X_train_reshaped, y_train_onehot, validation_data=(X_test_reshaped, y_test_onehot),
                        epochs=20, batch_size=16, verbose=1)

    # 模型评价与可视化
    y_pred = model.predict(X_test_reshaped)
    plot_evaluation(y_test, y_pred, history)

    # 三维聚类可视化
    visualize_clusters(features_scaled, labels, names)

    # 未知样品分类
'''
    classify_unknown_samples(model, scaler, unknown_file, le)
'''
