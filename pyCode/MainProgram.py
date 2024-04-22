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

'''
# 插值填补
print("中位数插值填补")
imp_mid = SimpleImputer(missing_values=np.NaN,strategy='median')
df_Impute = imp_mid.fit_transform(df)
display(df_Impute)
print("df.shape", df.shape)
np.savetxt("Temp/df_Impute.csv",df_Impute,'%.18e',delimiter=' ')
'''

# 提取Sample序列
print("提取序列")
firstList = xl_data['Sample']
print(firstList)
print(type(firstList))


#print("平均差与标准差")
#df_stats = df.describe().loc[['mean','std']]
#df_stats.style.format("{:.2f}")


# 归一化
scaler = StandardScaler()
df_Normal = scaler.fit_transform(df)
np.savetxt("Temp/Normalization.csv",df_Normal,'%.18e',delimiter=' ')


#降维
action_number = int(input("请输入数字选择降维方式：1.PCA降维 2.t-SNE降维"))
dimensionData = Main_Controller.dimension_Action(action_number, df_Normal, firstList)


'''
pca_Result = pca_moudle.do_Pca(df_Normal)
plt.title("PCA_Result")
plt.scatter(pca_Result[:, 0], pca_Result[:, 1])
plt.axis('equal')
for label,x,y in zip(firstList,pca_Result[:, 0],pca_Result[:, 1]):
    plt.text(x,y,label)
plt.show(block=False)
'''


# 组装pcaResult与firstList为DataFrame
df_firstList = pd.DataFrame({'pointName':firstList.values})
df_PcaLabelLocation = pd.DataFrame(dimensionData)
df_PcaLabelLocation.columns = ['scatter_X','scatter_Y']
df_PcaLabelLocation['Scatter_Index'] = df_firstList['pointName']
print(df_PcaLabelLocation)



#tSNE降维
#print("tSNE降维")
#tSNE_Result = tSNE_moudle.do_tSNE(df_Normal)

print("Kmeans聚类")
k_means_moudle.do_Kmeans(dimensionData,df_PcaLabelLocation)
#Kmeans_tSNE = k_means_moudle.do_Kmeans(tSNE_Result,firstList)


#print("KmeansConstrained聚类")
#Kmeans_constrained = kmeans_constrained_moudle.doKmeansConstrained(pca_Result,firstList)


