# Kernelized-KNN-and-K-means



# what is Kernel KNN?

Kernel k-nearest neighbors (Kernel KNN) is a machine learning algorithm that uses different weight functions (kernels) to optimize the output predictions in both regression and classification. It is an extension of the k-nearest neighbors algorithm (KNN), which is a non-parametric supervised learning method used for classification and regression.
In Kernel KNN, the input features are transformed into high-dimensional features using kernel methods. The proposed model KNN with kernel (K-KNN) improves the accuracy of classification. A novel reduced kernel method is also proposed and used in model K-KNN, which is named as Reduced Kernel KNN (RK-KNN). 

Details:

kernel KNN are delevoped by using 4 kernels, (RBF, Linear, PPOLYNOMIAL with degree = [1,2,3]) and then accuracy and F1 score are calculated on 4 different datasets of UCI, (Wine, Sonar, Glass and Breast). 

# what is Kernel K-means
Kernel k-means is an extension of the standard k-means algorithm that uses kernel functions to map the input data into a higher-dimensional space. This allows the algorithm to find non-linear decision boundaries between clusters.
The kernel k-means algorithm works by first computing a kernel matrix that measures the similarity between each pair of data points. Then, it applies the standard k-means algorithm to this kernel matrix instead of the original data.
Kernel k-means is useful when the data is not linearly separable in the original feature space. By mapping the data into a higher-dimensional space using a kernel function, it becomes more likely that the data will be linearly separable in this new space.


# kernelized PCA


Kernel PCA (Principal Component Analysis) is a nonlinear extension of PCA that uses kernel functions to project the data into a higher-dimensional space before performing PCA.
In traditional PCA, the data is projected onto a lower-dimensional subspace that captures the maximum amount of variance in the data. However, this approach only works well when the data is linearly separable. When the data is nonlinearly separable, kernel PCA can be used to project the data into a higher-dimensional space where it is more likely to be linearly separable.
Kernel PCA works by first computing a kernel matrix that measures the similarity between each pair of data points. Then, it applies PCA to this kernel matrix instead of the original data.

# details

different plots are used to compare different Kernels on train and test

 
![image](https://github.com/hamed-tgh/Kernelized-KNN-and-K-means/assets/47190471/232df7a6-7e8c-400a-9c76-faa9667a92ec)

 
 ![image](https://github.com/hamed-tgh/Kernelized-KNN-and-K-means/assets/47190471/f9d68df8-36ea-40a5-aedb-84394405ece6)

 ![image](https://github.com/hamed-tgh/Kernelized-KNN-and-K-means/assets/47190471/14df031b-24c2-4e2b-a73c-dda6bd39eb08)

 ![image](https://github.com/hamed-tgh/Kernelized-KNN-and-K-means/assets/47190471/5ec2c0ed-3ca2-4f35-835a-9ea23ade5af1)


