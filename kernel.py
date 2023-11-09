# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:48:07 2023

@author: 
"""




import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels


class Kernels:
    def __init__(self, XX, kernel , kernel_params = 1 ):
        self.X = XX.copy()
        self.k_opts = kernel_params
        self.kernel = kernel
    def process_data(self):

        ker = self.kernel
        XX = self.X

        if ker == "poly":
            X = self.poly(XX, gamma=1, degree=self.k_opts,
                          ).real
  
        elif ker == "rbf":
            X = self.rbf(XX, gamma=self.k_opts).real
        
        elif ker == "linear":
            X = self.linear(XX)



        return X





    def pca(self, X, n_components=2):

        (n, d) = X.shape
        X -= np.mean(X, 0)
        _cov = np.cov(X.T)
        U = self.eignes(_cov, n_components=n_components)
        XX = np.dot(X, U)
        return XX

    def poly(self, X, gamma=1, degree=2, n_components=2):

        X -= np.mean(X, 0)
        gamma = 1 / X.shape[1]
        K = (gamma*X.dot(X.T)+1)**degree

        return K

    def rbf(self, X, gamma=.1, n_components=2):
        X -= np.mean(X, 0)
        mat_sq_dists = np.sum((X[None, :] - X[:, None])**2, -1)
        K = np.exp(-gamma*mat_sq_dists)

        return K

    def linear(self , X ):
        return pairwise_kernels(X, X, metric=self.kernel,
                                filter_params=True, )
