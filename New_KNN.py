import numpy as np
from collections import Counter
from kernel import Kernels
import pandas as pd
import time
from sklearn.model_selection import train_test_split
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:48:07 2023

@author: 
"""




from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn import neighbors

def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))


class KNN:

    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in k_idx]  
        # return the most common class label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]


if __name__ == '__main__':
     
    # reading csv files
    wine =  pd.read_csv('Datasets//wine.data', sep=",")
    # print(wine)
    y_wine = wine.iloc[:,0]
    x_wine = wine.iloc[:,1:]
    wine = [x_wine.values , y_wine.values]


    sonar =  pd.read_csv('Datasets//sonar_all.data', sep=",")
    # print(sonar)
    shape = sonar.shape[1]
    y_sonar = sonar.iloc[:,-1]
    y_sonar = y_sonar.replace("R" , 0)
    y_sonar = y_sonar.replace("M" , 1)
    x_sonar = sonar.iloc[:, 0:shape-1]
    sonar = [x_sonar.values , y_sonar.values]

    sonarionosphere =  pd.read_csv('Datasets//ionosphere.data', sep=",")
    # print(sonarionosphere)
    shape = sonarionosphere.shape[1]
    sonarionosphere_y = sonarionosphere.iloc[:,-1]
    sonarionosphere_x = sonarionosphere.iloc[:,0:shape-1]
    sonarionosphere = [sonarionosphere_x.values , sonarionosphere_y.values ]

    glass =  pd.read_csv('Datasets//glass.data', sep=",")
    # print(glass)
    shape = glass.shape[1]
    glass_y = glass.iloc[:,-1]
    glass_x = glass.iloc[:,0:shape-1]
    glass = [glass_x.values , glass_y.values ]

    breast =  pd.read_excel('Datasets//BreastTissue.xls', sheet_name="Data")
    # print(breast)
    breast_y= breast['Class']
    breast_y = breast_y.replace('car' , 0)
    breast_y = breast_y.replace('fad' , 1)
    breast_y = breast_y.replace('mas' , 2)
    breast_y = breast_y.replace('gla' , 3)
    breast_y = breast_y.replace('con' , 4)
    breast_y = breast_y.replace('adi' , 5)
    breast_x = breast.drop(['Class'] , axis = 1 ).iloc[:, 3:5]
    breast = [breast_x.values , breast_y.values]


    h = .02


    counter_data = 0
    
    datasets_name = ["Wine" , "Sonar" , "Glass" , "Breast"]
    datasets = [wine , sonar, glass ,breast]
    kernels = ["rbf" , "poly", "poly", "poly","linear"]
    kernels_params = [1 , 1, 2,3,None]
    for data in datasets:
        print(datasets_name[counter_data])
        
        
    
    
        counter_kernel = 0 
        for ker in kernels:

            print(kernels[counter_kernel])
            kernel = Kernels(data[0] , kernels[counter_kernel] , kernels_params[counter_kernel])
            data_temp = kernel.process_data()


            X_train, X_test, y_train, y_test = train_test_split(
                    data[0], data[1], test_size=0.3)
            st = time.time()
            Knn=KNN()
            Knn.fit(X_train,y_train)
            yhat=Knn.predict(X_test)
            yhat_rbf = yhat
            KNN_accuracy_rbf = accuracy_score(y_test, yhat_rbf)
            KNN_f1_rbf = f1_score(y_test, yhat_rbf, average='macro')
            
            et = time.time()
            rbf_time = et - st
            print("KNN_"+kernels[counter_kernel]+" params "+ str(kernels_params[counter_kernel])
                  + " ACCU , F1 , EST_TIM" , KNN_accuracy_rbf , KNN_f1_rbf , rbf_time)
            counter_kernel += 1
        counter_data += 1
            
            
            
            
    
