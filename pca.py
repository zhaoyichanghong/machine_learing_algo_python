import numpy as np
import metrics

class PCA:
    def __init__(self, component_number, whiten=False, method='', visualize=False):
        self.__component_number = component_number
        self.__whiten = whiten
        self.__method = method
        self.__visualize = visualize

    def fit_transform(self, X):
        data_number = X.shape[0]

        self.__mean = np.mean(X, axis=0)
        X_sub_mean = X - self.__mean

        if self.__method == 'svd':
            u, s, vh = np.linalg.svd(X_sub_mean)
            self.__eig_values = (s ** 2)[:self.__component_number]
            self.__eig_vectors = vh.T[:, :self.__component_number]
        else:
            conv = X_sub_mean.T.dot(X_sub_mean)
            eig_values, eig_vectors = np.linalg.eigh(conv)
            self.__eig_values = eig_values[::-1][:self.__component_number]
            self.__eig_vectors = eig_vectors[:, ::-1][:, :self.__component_number]
        
        if self.__whiten:
            self.__std = np.sqrt(self.__eig_values.reshape((1, -1)) / (data_number - 1))

        return self.transform(X)
        
    def transform(self, X):
        X_sub_mean = X - self.__mean

        pc = X_sub_mean.dot(self.__eig_vectors)

        if self.__whiten:
            pc /= self.__std

        if self.__component_number == 2 and self.__visualize:
            metrics.scatter_feature(pc)

        return pc

class KernelPCA:
    def __init__(self, component_number, kernel_func, gamma=1, visualize=False):
        self.__component_number = component_number
        self.__kernel_func = kernel_func
        self.__gamma = gamma
        self.__visualize = visualize

    def fit_transform(self, X):
        self.__X = X
        data_number = self.__X.shape[0]
        
        K = self.__kernel_func(self.__X, self.__X, self.__gamma)
        self.__K_row_mean = np.mean(K, axis=0)
        self.__K_mean = np.mean(self.__K_row_mean)

        I = np.full((data_number, data_number), 1 / data_number)
        K_hat = K - I.dot(K) - K.dot(I) + I.dot(K).dot(I)

        eig_values, eig_vectors = np.linalg.eigh(K_hat)
        self.__eig_values = eig_values[::-1][:self.__component_number]
        self.__eig_vectors = eig_vectors[:, ::-1][:, :self.__component_number]

        pc = self.__eig_vectors * np.sqrt(self.__eig_values)

        if self.__component_number == 2 and self.__visualize:
            metrics.scatter_feature(pc)

        return pc
    
    def __kernel_centeralization(self, kernel):
        kernel -= self.__K_row_mean
        K_pred_cols = (np.sum(kernel, axis=1) / self.__K_row_mean.shape[0]).reshape((-1, 1))
        kernel -= K_pred_cols
        kernel += self.__K_mean

        return kernel

    def transform(self, X):
        kernel = self.__kernel_func(self.__X, X, self.__gamma)
        kernel = self.__kernel_centeralization(kernel)
        pc = kernel.dot(self.__eig_vectors / np.sqrt(self.__eig_values))

        if self.__component_number == 2 and self.__visualize:
            metrics.scatter_feature(pc)

        return pc

class ZCAWhiten:
    def __init__(self, method=''):
        self.__method = method
    
    def fit_transform(self, X):
        data_number = X.shape[0]

        self.__mean = np.mean(X, axis=0)
        X_sub_mean = X - self.__mean

        if self.__method == 'svd':
            u, s, vh = np.linalg.svd(X_sub_mean)
            self.__eig_values = s ** 2
            self.__eig_vectors = vh.T
        else:
            conv = X_sub_mean.T.dot(X_sub_mean)
            eig_values, eig_vectors = np.linalg.eigh(conv)
            self.__eig_values = eig_values[::-1]
            self.__eig_vectors = eig_vectors[:, ::-1]

        self.__std = np.sqrt(self.__eig_values.reshape((1, -1)) / (data_number - 1))

        return self.transform(X)

    def transform(self, X):
        X_sub_mean = X - self.__mean

        pc = X_sub_mean.dot(self.__eig_vectors)
        pc /= self.__std

        return pc.dot(self.__eig_vectors.T)