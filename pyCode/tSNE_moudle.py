from sklearn.manifold import TSNE

def do_tSNE(X):
    tsne = TSNE(n_components=2,perplexity=2,learning_rate=50)
    L = tsne.fit_transform(X)
    print(tsne.embedding_)
    return L
