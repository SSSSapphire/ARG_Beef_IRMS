import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def do_tSNE(data, firstList):
    tsne = TSNE(n_components=2,perplexity=2,learning_rate=50)
    #perplexity影响样品、簇间的间距；learning_rate越小，样品间距越小
    tSNE_Result = tsne.fit_transform(data)
    plt.title("tSNE_Result")
    plt.scatter(tSNE_Result[:, 0], tSNE_Result[:, 1])
    plt.axis('equal')
    print(tsne.embedding_)
    for label,x,y in zip(firstList,tSNE_Result[:, 0],tSNE_Result[:, 1]):
        plt.text(x,y,label)
    plt.show(block=False)
    return tSNE_Result
