import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

# 加载数据
def load_data(file_path):
    df = pd.read_csv(file_path)
    names = df.iloc[:, 0].values  # 样品名称
    labels = df.iloc[:, 1].values  # 类别标签
    data = df.iloc[:, 2:].values  # 变量
    return names, labels, data

# 数据预处理
def preprocess_data(labels, data):
    # 编码字符串标签为整数
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    # 标准化数据
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    # 转为独热编码
    labels_onehot = to_categorical(labels_encoded)
    return label_encoder, scaler, data_scaled, labels_onehot

# 建立CNN模型
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
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 二维降维并绘制聚类图
def plot_2d_tsne(data, labels, names):
    tsne = TSNE(n_components=2, random_state=42)
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        indices = labels == label
        if label==1 :
            label_name = "CHN"
        else:
            label_name = "ARG"
        plt.scatter(data_2d[indices, 0], data_2d[indices, 1], label=f'Class {label_name}')
    '''for i, name in enumerate(names):
        plt.text(data_2d[i, 0], data_2d[i, 1], name, fontsize=8)'''
    plt.legend()
    plt.title("PCA Projection")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid()
    plt.show()
    
def plot_3d_pca(data, labels, names):
    tsne = TSNE(n_components=3, random_state=42)
    pca = PCA(n_components=3)
    data_3d = pca.fit_transform(data)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        indices = labels == label
        if label == 1:
            label_name = "CHN"
        else:
            label_name = "ARG"
        ax.scatter(
            data_3d[indices, 0], 
            data_3d[indices, 1], 
            data_3d[indices, 2], 
            label=f'Class {label_name}', 
            alpha=0.8
        )
    # 添加样本名称
    '''
    for i, name in enumerate(names):
        ax.text(
            data_3d[i, 0], 
            data_3d[i, 1], 
            data_3d[i, 2], 
            name, 
            fontsize=8
        )
    '''
    # 设置标题和坐标轴标签
    ax.set_title("3D PCA Projection")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    
    # 添加图例和网格
    ax.legend()
    plt.grid()
    plt.show()

# 预测未知样品
def predict_unknown_samples(model, scaler, label_encoder, file_path):
    df = pd.read_csv(file_path)
    names = df.iloc[:, 0].values  # 样品名称
    data = df.iloc[:, 2:].values  # 变量
    data_scaled = scaler.transform(data)
    data_scaled = data_scaled[..., np.newaxis]  # 添加维度以适配CNN输入
    predictions = model.predict(data_scaled)
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_labels = label_encoder.inverse_transform(predicted_classes)
    results = pd.DataFrame({
        'Sample Name': names,
        'Predicted Class': predicted_labels,
        'Confidence': np.max(predictions, axis=1)
    })
    return results

# 主程序
if __name__ == "__main__":
    # 配置文件路径
    file_path = "Sci01_Data_CNN_Train.csv"  # 样品数据
    unknown_file_path = "Sci01_Data_CNN_Test.csv"  # 未知样品数据
    output_file = "predicted_results.csv"  # 预测结果保存路径

    # 加载和预处理数据
    names, labels, data = load_data(file_path)
    label_encoder, scaler, data_scaled, labels_onehot = preprocess_data(labels, data)

    # 划分数据集
    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        data_scaled, labels_onehot, names, test_size=0.2, random_state=42)
    X_train_cnn = X_train[..., np.newaxis]  # 为CNN添加维度
    X_test_cnn = X_test[..., np.newaxis]

    # 建立并训练CNN模型
    input_shape = (X_train_cnn.shape[1], X_train_cnn.shape[2])
    num_classes = y_train.shape[1]
    cnn_model = create_cnn_model(input_shape, num_classes)
    cnn_model.fit(X_train_cnn, y_train, epochs=20, batch_size=8, validation_split=0.2, verbose=1)

    # 保存模型和预处理器
    cnn_model.save("cnn_model.h5")
    np.save("scaler_mean.npy", scaler.mean_)
    np.save("scaler_scale.npy", scaler.scale_)
    with open("label_classes.npy", "wb") as f:
        np.save(f, label_encoder.classes_)

    # 测试模型
    test_accuracy = cnn_model.evaluate(X_test_cnn, y_test, verbose=0)[1]
    print(f"Test Accuracy: {test_accuracy:.2%}")

    # 二维降维并绘图
    plot_2d_tsne(data_scaled, label_encoder.transform(labels), names)
    plot_3d_pca(data_scaled, label_encoder.transform(labels), names)
    # 预测未知样品
    results = predict_unknown_samples(cnn_model, scaler, label_encoder, unknown_file_path)
    results.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
