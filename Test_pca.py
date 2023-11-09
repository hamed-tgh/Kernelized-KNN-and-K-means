# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 17:40:46 2023

@author: hamed
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


    
    from sklearn.decomposition import PCA, KernelPCA

    pca = PCA(n_components=2)
    kernel_pca = KernelPCA(
        n_components=None, kernel="rbf", gamma=10, fit_inverse_transform=True, alpha=0.1
    )
    
    X_test_pca = pca.fit(X_train2).transform(X_test2)
    X_test_kernel_pca = kernel_pca.fit(X_train2).transform(X_test2)
    
    
    fig, (orig_data_ax, pca_proj_ax, kernel_pca_proj_ax) = plt.subplots(
    ncols=3, figsize=(14, 4)
    )

    orig_data_ax.scatter(X_test2[:, 0], X_test2[:, 1], c=y_test2)
    orig_data_ax.set_ylabel("Feature #1")
    orig_data_ax.set_xlabel("Feature #0")
    orig_data_ax.set_title("Testing data")
    
    pca_proj_ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test2)
    pca_proj_ax.set_ylabel("Principal component #1")
    pca_proj_ax.set_xlabel("Principal component #0")
    pca_proj_ax.set_title("Projection of testing data\n using PCA")
    
    kernel_pca_proj_ax.scatter(X_test_kernel_pca[:, 0], X_test_kernel_pca[:, 1], c=y_test2)
    kernel_pca_proj_ax.set_ylabel("Principal component #1")
    kernel_pca_proj_ax.set_xlabel("Principal component #0")
    _ = kernel_pca_proj_ax.set_title("Projection of testing data\n using KernelPCA")

