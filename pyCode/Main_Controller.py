import pca_moudle
import tSNE_moudle

def dimension_Action(action_number, data, firstList):
    if action_number == 1:
        print("PCA降维")
        return pca_moudle.do_Pca(data, firstList)
    elif action_number == 2:
        print("t-SNE降维")
        return tSNE_moudle.do_tSNE(data, firstList)