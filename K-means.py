# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:48:07 2023

@author: 
"""




import seaborn as sns
from sklearn.datasets import make_moons , make_circles
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_random_state
from kernel import Kernels as ker
import numpy as np
from scipy.spatial.distance import cdist 
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

"""
    Kernel K-means
    
    Reference
    ---------
    Kernel k-means, Spectral Clustering and Normalized Cuts.
    Inderjit S. Dhillon, Yuqiang Guan, Brian Kulis.
    KDD 2004.
    """


class KernelKMeans(BaseEstimator, ClusterMixin):
    

    def __init__(self, n_clusters=3, max_iter=50, tol=1e-3, random_state=None,
                 kernel="linear", gamma=None, degree=3, coef0=1,
                 kernel_params=None, verbose=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.verbose = verbose
        self.watch = False
    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def _get_kernel(self, X, Y=None):
         if self.kernel == "poly":
            degree = 1
            kernel = ker(X, 'poly' , kernel_params= self.kernel_params)
            XX = kernel.process_data()
            return XX
         
         elif self.kernel == "rbf":
            gamma =  1
            kernel = ker(X, 'rbf' , kernel_params= self.kernel_params)
            XX = kernel.process_data()
            return XX
            


    def fit(self,x , x2):
        

        idx = np.random.choice(len(x), self.n_clusters, replace=False)
        #Randomly choosing Centroids 
        self.centroids = x[idx, :] #Step 1
        
        #finding the distance between centroids and all the data points
        distances = cdist(x, self.centroids ,'euclidean') #Step 2
        
        #Centroid with the minimum Distance
        points = np.array([np.argmin(i) for i in distances]) #Step 3
        
        #Repeating the above steps for a defined number of iterations
        #Step 4
        for _ in range(self.max_iter): 
            self.centroids = []
            for idx in range(self.n_clusters):
                #Updating Centroids by taking mean of Cluster it belongs to
                temp_cent = x[points==idx].mean(axis=0) 
                self.centroids.append(temp_cent)
    
            self.centroids = np.vstack(self.centroids) #Updated Centroids 
            
            distances = cdist(x, self.centroids ,'euclidean')
            points = np.array([np.argmin(i) for i in distances])
            #dispaly points and coresepond label
            if self.watch == True:
                u_labels = np.unique(points)
                for i in u_labels:
                    plt.scatter(x2[points == i , 0] , x2[points == i , 1] , label = i)
                plt.legend()
                plt.show()  
        return points 
    


    def predict(self, X):
        distances = cdist(X, self.centroids ,'euclidean')
        points = np.array([np.argmin(i) for i in distances])
        return points
    

if __name__ == '__main__':


    print('"""""""""""""""make moooon resultss""""""""""""""""""')
    ##############make moon ########################
    X_train_org, true_labels = make_moons(n_samples=100, noise=0.05, random_state=0)
    #X_train = StandardScaler().fit_transform(X_train)
    x_train2, x_test2 , y_train2 , y_test2 = train_test_split(X_train_org,true_labels, test_size=0.3, shuffle=False)

    sns.scatterplot(x=[X[0] for X in X_train_org],
                    y=[X[1] for X in X_train_org],
                    hue=true_labels,
                    palette="deep",
                    legend=None
                    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("the data set over view")
    plt.show()




    #RBF-kernel
    print("############RBF kernel is running################")
    km = KernelKMeans(n_clusters=2, max_iter=10, random_state=0, verbose=1 , kernel="rbf" , kernel_params=1/100)
    print("the shape of X before doing kernel" , X_train_org.shape)
    X_train2 = km._get_kernel(X_train_org)
    #X_train2 = (X_train) #if you want to run your code without kernels you can just decomment it
    print("the shape of X after doing kernel" , X_train2.shape)

    #train_test_validation
    x_train, x_test , y_train , y_test = train_test_split(X_train2,true_labels, test_size=0.3, shuffle=False)

    print(x_train.shape)
    print(x_test.shape)
    _ = km.fit(x_train , x_train2)
    label = km.predict(x_test)
    u_labels = np.unique(label)
    for i in u_labels:
        plt.scatter(x_test2[label == i , 0] , x_test2[label == i , 1] , label = i)
    plt.legend()
    plt.title("the result of RBF")
    plt.show()
    
    print("the accuracy is : " , accuracy_score(label , y_test))
    print("the F1 score is : " , f1_score(label , y_test))
    print("the precision score is : " , precision_score(label , y_test))
    print("the recall score is : " , recall_score(label , y_test))



    #POLY is running with degree = 1
    print("##########POLY kernel is running with degree 1#############")
    km = KernelKMeans(n_clusters=2, max_iter=10, random_state=0, verbose=1 , kernel="poly" , kernel_params=1)
    print("the shape of X before doing kernel" , X_train_org.shape)
    X_train2 = km._get_kernel(X_train_org)
    #X_train2 = (X_train) #if you want to run your code without kernels you can just decomment it
    print("the shape of X after doing kernel" , X_train2.shape)

    #train_test_validation
    x_train, x_test , y_train , y_test = train_test_split(X_train2,true_labels, test_size=0.3, shuffle=False)

    print(x_train.shape)
    print(x_test.shape)
    _ = km.fit(x_train , x_train2)
    label = km.predict(x_test)
    u_labels = np.unique(label)
    for i in u_labels:
        plt.scatter(x_test2[label == i , 0] , x_test2[label == i , 1] , label = i)
    plt.legend()
    plt.title("the result of poly with degree 1")
    plt.show()
    print("the accuracy is : " , accuracy_score(label , y_test))
    print("the F1 score is : " , f1_score(label , y_test))
    print("the precision score is : " , precision_score(label , y_test))
    print("the recall score is : " , recall_score(label , y_test))



#POLY is running with degree = 2
    print("#############POLY kernel is running with degree 3##############")
    km = KernelKMeans(n_clusters=2, max_iter=10, random_state=0, verbose=1 , kernel="poly" , kernel_params=2)
    print("the shape of X before doing kernel" , X_train_org.shape)
    X_train2 = km._get_kernel(X_train_org)
    #X_train2 = (X_train) #if you want to run your code without kernels you can just decomment it
    print("the shape of X after doing kernel" , X_train2.shape)

    #train_test_validation
    x_train, x_test , y_train , y_test = train_test_split(X_train2,true_labels, test_size=0.3, shuffle=False)

    print(x_train.shape)
    print(x_test.shape)
    _ = km.fit(x_train , x_train2)
    label = km.predict(x_test)
    u_labels = np.unique(label)
    for i in u_labels:
        plt.scatter(x_test2[label == i , 0] , x_test2[label == i , 1] , label = i)
    plt.legend()
    plt.title("the result of poly with degree 2")
    plt.show()
    print("the accuracy is : " , accuracy_score(label , y_test))
    print("the F1 score is : " , f1_score(label , y_test))
    print("the precision score is : " , precision_score(label , y_test))
    print("the recall score is : " , recall_score(label , y_test))

#POLY is running with degree = 3
    print("#################POLY kernel is running with degree 3#####################")
    km = KernelKMeans(n_clusters=2, max_iter=10, random_state=0, verbose=1 , kernel="poly" , kernel_params=3)
    print("the shape of X before doing kernel" , X_train_org.shape)
    X_train2 = km._get_kernel(X_train_org)
    #X_train2 = (X_train) #if you want to run your code without kernels you can just decomment it
    print("the shape of X after doing kernel" , X_train2.shape)

    #train_test_validation
    x_train, x_test , y_train , y_test = train_test_split(X_train2,true_labels, test_size=0.3, shuffle=False)

    print(x_train.shape)
    print(x_test.shape)
    _ = km.fit(x_train , x_train2)
    label = km.predict(x_test)
    u_labels = np.unique(label)
    for i in u_labels:
        plt.scatter(x_test2[label == i , 0] , x_test2[label == i , 1] , label = i)
    plt.legend()
    plt.title("the result of poly with degree 3")
    plt.show()
    print("the accuracy is : " , accuracy_score(label , y_test))
    print("the F1 score is : " , f1_score(label , y_test))
    print("the precision score is : " , precision_score(label , y_test))
    print("the recall score is : " , recall_score(label , y_test))



    print()
    print('##############make circle##################')
    print()
    X_train_org, true_labels = make_circles(n_samples=100, factor=0.3, noise=0.05, random_state=0)
    #X_train = StandardScaler().fit_transform(X_train)
    x_train2, x_test2 , y_train2 , y_test2 = train_test_split(X_train_org,true_labels, test_size=0.3, shuffle=False)

    sns.scatterplot(x=[X[0] for X in X_train_org],
                    y=[X[1] for X in X_train_org],
                    hue=true_labels,
                    palette="deep",
                    legend=None
                    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("the over view of dataset")
    plt.show()




    #RBF-kernel
    print("############RBF kernel is running################")
    km = KernelKMeans(n_clusters=2, max_iter=10, random_state=0, verbose=1 , kernel="rbf" , kernel_params=1/500)
    print("the shape of X before doing kernel" , X_train_org.shape)
    X_train2 = km._get_kernel(X_train_org)
    #X_train2 = (X_train) #if you want to run your code without kernels you can just decomment it
    print("the shape of X after doing kernel" , X_train2.shape)

    #train_test_validation
    x_train, x_test , y_train , y_test = train_test_split(X_train2,true_labels, test_size=0.3, shuffle=False)

    print(x_train.shape)
    print(x_test.shape)
    _ = km.fit(x_train , x_train2)
    label = km.predict(x_test)
    u_labels = np.unique(label)
    for i in u_labels:
        plt.scatter(x_test2[label == i , 0] , x_test2[label == i , 1] , label = i)
    plt.legend()
    plt.title("the result of RBF")
    plt.show()
    print("the accuracy is : " , accuracy_score(label , y_test))
    print("the F1 score is : " , f1_score(label , y_test))
    print("the precision score is : " , precision_score(label , y_test))
    print("the recall score is : " , recall_score(label , y_test))


    #POLY is running with degree = 1
    print("##########POLY kernel is running with degree 1#############")
    km = KernelKMeans(n_clusters=2, max_iter=10, random_state=0, verbose=1 , kernel="poly" , kernel_params=1)
    print("the shape of X before doing kernel" , X_train_org.shape)
    X_train2 = km._get_kernel(X_train_org)
    #X_train2 = (X_train) #if you want to run your code without kernels you can just decomment it
    print("the shape of X after doing kernel" , X_train2.shape)

    #train_test_validation
    x_train, x_test , y_train , y_test = train_test_split(X_train2,true_labels, test_size=0.3, shuffle=False)

    print(x_train.shape)
    print(x_test.shape)
    _ = km.fit(x_train , x_train2)
    label = km.predict(x_test)
    u_labels = np.unique(label)
    for i in u_labels:
        plt.scatter(x_test2[label == i , 0] , x_test2[label == i , 1] , label = i)
    plt.legend()
    plt.title("the result of poly with degree 1")
    plt.show()
    print("the accuracy is : " , accuracy_score(label , y_test))
    print("the F1 score is : " , f1_score(label , y_test))
    print("the precision score is : " , precision_score(label , y_test))
    print("the recall score is : " , recall_score(label , y_test))



#POLY is running with degree = 2
    print("#############POLY kernel is running with degree 3##############")
    km = KernelKMeans(n_clusters=2, max_iter=10, random_state=0, verbose=1 , kernel="poly" , kernel_params=2)
    print("the shape of X before doing kernel" , X_train_org.shape)
    X_train2 = km._get_kernel(X_train_org)
    #X_train2 = (X_train) #if you want to run your code without kernels you can just decomment it
    print("the shape of X after doing kernel" , X_train2.shape)

    #train_test_validation
    x_train, x_test , y_train , y_test = train_test_split(X_train2,true_labels, test_size=0.3, shuffle=False)

    print(x_train.shape)
    print(x_test.shape)
    _ = km.fit(x_train , x_train2)
    label = km.predict(x_test)
    u_labels = np.unique(label)
    for i in u_labels:
        plt.scatter(x_test2[label == i , 0] , x_test2[label == i , 1] , label = i)
    plt.legend()
    plt.title("the result of poly with degree 2")
    plt.show()
    print("the accuracy is : " , accuracy_score(label , y_test))
    print("the F1 score is : " , f1_score(label , y_test))
    print("the precision score is : " , precision_score(label , y_test))
    print("the recall score is : " , recall_score(label , y_test))

#POLY is running with degree = 3
    print("#################POLY kernel is running with degree 3#####################")
    km = KernelKMeans(n_clusters=2, max_iter=10, random_state=0, verbose=1 , kernel="poly" , kernel_params=3)
    print("the shape of X before doing kernel" , X_train_org.shape)
    X_train2 = km._get_kernel(X_train_org)
    #X_train2 = (X_train) #if you want to run your code without kernels you can just decomment it
    print("the shape of X after doing kernel" , X_train2.shape)

    #train_test_validation
    x_train, x_test , y_train , y_test = train_test_split(X_train2,true_labels, test_size=0.3, shuffle=False)

    print(x_train.shape)
    print(x_test.shape)
    _ = km.fit(x_train , x_train2)
    label = km.predict(x_test)
    u_labels = np.unique(label)
    for i in u_labels:
        plt.scatter(x_test2[label == i , 0] , x_test2[label == i , 1] , label = i)
    plt.legend()
    plt.title("the result of poly with degree 3")

    plt.show()
    print("the accuracy is : " , accuracy_score(label , y_test))
    print("the F1 score is : " , f1_score(label , y_test))
    print("the precision score is : " , precision_score(label , y_test))
    print("the recall score is : " , recall_score(label , y_test))
