import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

n_components = 2
random_state = 9527
pca = PCA(n_components=n_components, 
          random_state=random_state)
def do_Pca(data, firstList):
    pca_2d = PCA(n_components, random_state=random_state)
    pca_Result = pca_2d.fit_transform(data)
    plt.title("PCA_Result")
    plt.scatter(pca_Result[:, 0], pca_Result[:, 1])
    plt.axis('equal')
    for label,x,y in zip(firstList,pca_Result[:, 0],pca_Result[:, 1]):
        plt.text(x,y,label)
    plt.show(block=False)
    return pca_Result




    