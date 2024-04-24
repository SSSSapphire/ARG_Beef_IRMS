import PCA_moudle
import tSNE_moudle
import kPCA_moudle
import k_means_moudle
import sys
import pandas as pd

def dimension_Action(inputNumber, data, firstList):
    if inputNumber == 1:
        print("PCA降维")
        return PCA_moudle.do_Pca(data, firstList)
    elif inputNumber == 2:
        print("t-SNE降维")
        return tSNE_moudle.do_tSNE(data, firstList)
    elif inputNumber == 3:
        print("kPCA降维")
        return kPCA_moudle.do_kPCA(data, firstList)
    
def DataChoice(inputNumber):
    if inputNumber == 1:
        df = pd.read_csv('Data/name_beef_elements_0424.csv',index_col="Sample")
        print("降维分析")
        return df
    elif inputNumber == 2:
        df = pd.read_csv('Data/beef_elements_0416.csv',index_col="Sample")
        print("聚类分析")
        return df

def clusteringMethod(inputNumber,data,firstList):
    if inputNumber == 1:
        print("K-means聚类")
        k_means_moudle.do_Kmeans(data,firstList)
    else:
        sys.exit()


    