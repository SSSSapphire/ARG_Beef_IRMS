import csv
import pandas as pd
import array as arr
import k_means_moudle
import tSNE_moudle
import Main_Controller
import kmeans_constrained_moudle
import numpy as np
import matplotlib.pyplot as plt
from operator import index
from tarfile import PAX_NAME_FIELDS
from IPython.display import display
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer    


# 读csv
xl_data = pd.read_csv('Data/beef_elements_0416.csv')


# 读取csv文件存为df
df = pd.read_csv('Data/beef_elements_0416.csv',index_col="Sample")
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
action_number = int(input("请输入数字选择降维方式：1.PCA降维 2.t-SNE降维 3.kPCA降维"))
dimensionData = Main_Controller.dimension_Action(action_number, df_Normal, firstList)


# 组装pcaResult与firstList为DataFrame
df_firstList = pd.DataFrame({'pointName':firstList.values})
df_PcaLabelLocation = pd.DataFrame(dimensionData)
df_PcaLabelLocation.columns = ['scatter_X','scatter_Y']
df_PcaLabelLocation['Scatter_Index'] = df_firstList['pointName']
print(df_PcaLabelLocation)


print("Kmeans聚类")
k_means_moudle.do_Kmeans(dimensionData,df_PcaLabelLocation)



