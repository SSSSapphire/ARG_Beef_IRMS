from sklearn.decomposition import PCA

n_components = 2
random_state = 9527
pca = PCA(n_components=n_components, 
          random_state=random_state)
def do_Pca(X):
    pca_2d = PCA(n_components, random_state=random_state)
    L = pca_2d.fit_transform(X)
    return L



    