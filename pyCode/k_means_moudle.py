import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

def do_Kmeans(X,df_Location):
    n_cluster = 2
    random_state = 0
    cluster =  KMeans(n_clusters = n_cluster, n_init= "auto",random_state = random_state).fit(X)
    print(cluster)

    #查看每个样本对应的类
    pred = cluster.labels_
    pred

    #使用部分数据预测质心
    pre = cluster.fit_predict(X)
    pre == pred

    #质心
    centroid = cluster.cluster_centers_
    centroid
    centroid.shape

    #总距离平方和
    inertia = cluster.inertia_
    inertia

    #评价系数计算
    print("轮廓系数silhouette_score(0-1，越高越好) = " +              str(silhouette_score(X,pred)))
    print("卡林斯基哈拉巴斯指数calinski_harabasz_score(越高越好) = " + str(calinski_harabasz_score(X,pred)))
    print("戴维斯布尔丁指数davies_bouldin_score(越小越好) = " +        str(davies_bouldin_score(X,pred)))

    #预测结果分类标签名称
    df_Pred = pd.DataFrame({'pred_index':pred})
    df_Location['pred_index'] = df_Pred['pred_index']
    df_cluster1 = df_Location[df_Location.pred_index < 1]
    df_cluster2 = df_Location[df_Location.pred_index > 0]
    print(df_cluster1)
    print(df_cluster2)

    color = ["red","blue"]
    fig, ax1 = plt.subplots(1)
    tempCounter = 0
    for i in range(n_cluster):
        print("xxxxxxxxxxxxxxxxxxx", n_cluster,i,pre)
        ax1.scatter(X[pred==i, 0], X[pred==i, 1]
           ,marker='o'
           ,s=8
           ,c=color[i]
           )
        if(i==0):
            for label,x,y in zip(df_cluster1.Scatter_Index,X[pred==0, 0],X[pred==0, 1]):
                plt.text(x,y,label)
        if(i==1):
            for label,x,y in zip(df_cluster2.Scatter_Index,X[pred==1, 0],X[pred==1, 1]):
                plt.text(x,y,label)
    ax1.scatter(centroid[:,0],centroid[:,1]
        ,marker="x"
        ,s=15
        ,c="black")
    plt.axis('equal')
    plt.title('Kmeans_result')
    plt.show()
    
    #因实验本身的目的，通过改变n_cluster分簇数量来影响inertia来评估分类数量效果不好
    #轮廓系数silhouette_score对聚类的评估有一定参考性
    #卡林斯基哈拉巴斯指数calinski_harabasz_score，优点快
    #戴维斯布尔丁指数davies_bouldin_score，优点：评价准确度相对高，稳定性好；缺点：较慢，对质点的选择敏感
    #对新加入的数据进行预测，以已有的数据所构建的模型
    #kmeans.predict([[0, 0], [1, 1]])