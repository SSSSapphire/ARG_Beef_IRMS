import csv
import pandas as pd
import array as arr
import k_means_moudle
import tSNE_moudle
import Main_Controller
import kmeans_constrained_moudle
import PCA_Location_moudle
import numpy as np
import matplotlib.pyplot as plt
from operator import index
from tarfile import PAX_NAME_FIELDS
from IPython.display import display
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer    


# 读csv
xl_data = pd.read_csv('Data/name_beef_elements_0424.csv')


# 选择分析方式
inputNumbe_A = int(input("请输入数字选择分析方式：\n 1.降维分析 \n 2.聚类分析"))
df = Main_Controller.DataChoice(inputNumbe_A)

# 读取csv文件存为df
np.savetxt("Temp/transCSV.csv",df,'%.18e',delimiter=' ')
print("原始数据:")
display(df)


# 提取Sample序列
print("提取序列")
firstList = xl_data['Sample']
print(firstList)
print(type(firstList))


# 归一化
scaler = StandardScaler()
df_Normal = scaler.fit_transform(df)
np.savetxt("Temp/Normalization.csv",df_Normal,'%.18e',delimiter=' ')


#降维
inputNumber_D = int(input("请输入数字选择降维方式：\n1.PCA降维\n2.t-SNE降维\n3.kPCA降维"))
dimensionData = Main_Controller.dimension_Action(inputNumber_D, df_Normal, firstList)


# 组装pcaResult与firstList为DataFrame
df_firstList = pd.DataFrame({'pointName':firstList.values})
df_PcaLabelLocation = pd.DataFrame(dimensionData)
df_PcaLabelLocation.columns = ['scatter_X','scatter_Y']
df_PcaLabelLocation['Scatter_Index'] = df_firstList['pointName']
print(df_PcaLabelLocation)
PCA_Location_moudle.Location_draw(df_PcaLabelLocation)

inputNumber_D = int(input("请输入数字选择聚类方式：1.k-means聚类"))
Main_Controller.clusteringMethod(inputNumber_D,dimensionData,df_PcaLabelLocation)




