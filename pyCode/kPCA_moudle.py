import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA

def do_kPCA(data, firstList):
    kPCA = KernelPCA(n_components= 2, kernel= "rbf",  gamma=10, fit_inverse_transform=True, alpha=0.1)
    kPCA_Result = kPCA.fit(data)
    plt.title("tSNE_Result")
    plt.scatter(kPCA_Result[:, 0], kPCA_Result[:, 1])
    plt.axis('equal')
    for label,x,y in zip(firstList,kPCA_Result[:, 0],kPCA_Result[:, 1]):
        plt.text(x,y,label)
    plt.show(block=False)
    return kPCA_Result
