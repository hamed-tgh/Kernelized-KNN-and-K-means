# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:48:07 2023

@author: 
"""



from scipy.spatial.distance import pdist , squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np
from sklearn.datasets import make_circles
from kernel import Kernels
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import numpy as np
import copy 
 


class PCA():
    def __init__(self , num_components ):
        
        self.num_components = num_components
    
    def fit(self , X):
        #Step-1
        X_meaned = X - np.mean(X , axis = 0)
         
        #Step-2
        cov_mat = np.cov(X_meaned , rowvar = False)
         
        #Step-3
        self.eigen_values , self.eigen_vectors = np.linalg.eigh(cov_mat)
        
        #Step-4
        sorted_index = np.argsort(self.eigen_values)[::-1]
        self.sorted_eigenvalue = self.eigen_values[sorted_index]
        self.sorted_eigenvectors = self.eigen_vectors[:,sorted_index]

        #Step-5
        self.eigenvector_subset = self.sorted_eigenvectors[:,0:self.num_components]
        
    def process(self , X):
        
        X_meaned = X - np.mean(X , axis = 0)
        X_reduced = np.dot(self.eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
        
        return X_reduced






if __name__ == '__main__':
    
    X_train, true_labels = make_circles(n_samples=1_000, factor=0.3, noise=0.05, random_state=0)   
    kernels = ["rbf" , "poly", "poly", "poly","linear"]
    kernels_params = [1 , 1, 2,3,None]
    counter_kernel = 0 
    
    X_train2, X_test2, y_train2, y_test2 = train_test_split(
            X_train, true_labels, test_size=0.3)
    _, (train_ax, test_ax) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8, 4))

    train_ax.scatter(X_train2[:, 0], X_train2[:, 1], c=y_train2)
    train_ax.set_ylabel("Feature #1")
    train_ax.set_xlabel("Feature #0")
    train_ax.set_title("Training data")
    
    test_ax.scatter(X_test2[:, 0], X_test2[:, 1], c=y_test2)
    test_ax.set_xlabel("Feature #0")
    _ = test_ax.set_title("Testing data")
    
    
    
        
    for ker in kernels:

        print(kernels[counter_kernel])
        kernel = Kernels(copy.copy(X_train) , kernels[counter_kernel] , kernels_params[counter_kernel])
        X_train_temp = kernel.process_data()


        X_train3, X_test3, y_train3, y_test3 = train_test_split(
                X_train_temp, true_labels, test_size=0.3)
        
    
              
        
        X_kpca = PCA(num_components=2)
        
        
        X_kpca.fit(X_train3)
        X_test_kernel_pca = X_kpca.process(X_test3)
        
        
        fig, (orig_data_ax, kernel_pca_proj_ax) = plt.subplots(
            ncols=2, figsize=(14, 4)
        )
        
        orig_data_ax.scatter(X_test2[:, 0], X_test2[:, 1], c=y_test2)
        orig_data_ax.set_ylabel("Feature #1")
        orig_data_ax.set_xlabel("Feature #0")
        orig_data_ax.set_title("Testing data")
        

        
        kernel_pca_proj_ax.scatter(X_test_kernel_pca[:, 0], X_test_kernel_pca[:, 1], c=y_test3)
        kernel_pca_proj_ax.set_ylabel("Principal component #1")
        kernel_pca_proj_ax.set_xlabel("Principal component #0")
        _ = kernel_pca_proj_ax.set_title("Projection of testing data\n using KernelPCA"+
                                         kernels[counter_kernel])
        counter_kernel+=1
        
        
        

    