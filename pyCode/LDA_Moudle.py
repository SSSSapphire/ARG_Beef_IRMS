import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

n_components = 2
def do_LDA(data, firstList):
    lda = LinearDiscriminantAnalysis(n_components)
    lda_Result = lda.fit(data)
    plt.title("LDA_Result")
    plt.scatter(lda_Result[:, 0], lda_Result[:, 1])
    plt.axis('equal')
    for label,x,y in zip(firstList,lda_Result[:, 0],lda_Result[:, 1]):
        plt.text(x,y,label)
    plt.show(block=False)
    return lda_Result
