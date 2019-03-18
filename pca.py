import numpy as np
import metrics

class PCA:
    def __init__(self, n_components, whiten=False, method='', visualize=False):
        '''
        Parameters
        ----------
        n_components : Number of components to keep
        whiten : Whitening
        method : SVD or not
        visualize : Plot scatter if n_components equals 2
        '''
        self.__n_components = n_components
        self.__whiten = whiten
        self.__method = method
        self.__visualize = visualize

    def fit_transform(self, X):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data

        Returns
        -------
        X : shape (n_samples, n_components)
            The data of dimensionality reduction
        '''
        n_samples = X.shape[0]

        self.__mean = np.mean(X, axis=0)
        X_sub_mean = X - self.__mean

        if self.__method == 'svd':
            u, s, vh = np.linalg.svd(X_sub_mean)
            self.__eig_values = (s ** 2)[:self.__n_components]
            self.__eig_vectors = vh.T[:, :self.__n_components]
        else:
            conv = X_sub_mean.T.dot(X_sub_mean)
            eig_values, eig_vectors = np.linalg.eigh(conv)
            self.__eig_values = eig_values[::-1][:self.__n_components]
            self.__eig_vectors = eig_vectors[:, ::-1][:, :self.__n_components]
        
        if self.__whiten:
            self.__std = np.sqrt(self.__eig_values.reshape((1, -1)) / (n_samples - 1))

        return self.transform(X)
        
    def transform(self, X):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Predicting data

        Returns
        -------
        X : shape (n_samples, n_components)
            The data of dimensionality reduction
        '''
        X_sub_mean = X - self.__mean

        pc = X_sub_mean.dot(self.__eig_vectors)

        if self.__whiten:
            pc /= self.__std

        if self.__n_components == 2 and self.__visualize:
            metrics.scatter_feature(pc)

        return pc

class KernelPCA:
    def __init__(self, n_components, kernel_func, sigma=1, visualize=False):
        '''
        Parameters
        ----------
        n_components : Number of components to keep
        kernel_func : kernel algorithm see also kernel.py
        sigma : Parameter for rbf kernel
        visualize : Plot scatter if n_components equals 2
        '''
        self.__n_components = n_components
        self.__kernel_func = kernel_func
        self.__sigma = sigma
        self.__visualize = visualize

    def fit_transform(self, X):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data

        Returns
        -------
        X : shape (n_samples, n_components)
            The data of dimensionality reduction
        '''
        self.__X = X
        n_samples = self.__X.shape[0]
        
        K = self.__kernel_func(self.__X, self.__X, self.__sigma)
        self.__K_row_mean = np.mean(K, axis=0)
        self.__K_mean = np.mean(self.__K_row_mean)

        I = np.full((n_samples, n_samples), 1 / n_samples)
        K_hat = K - I.dot(K) - K.dot(I) + I.dot(K).dot(I)

        eig_values, eig_vectors = np.linalg.eigh(K_hat)
        self.__eig_values = eig_values[::-1][:self.__n_components]
        self.__eig_vectors = eig_vectors[:, ::-1][:, :self.__n_components]

        #pc = self.__eig_vectors * np.sqrt(self.__eig_values)

        return self.transform(X)
    
    def __kernel_centeralization(self, kernel):
        kernel -= self.__K_row_mean
        K_pred_cols = (np.sum(kernel, axis=1) / self.__K_row_mean.shape[0]).reshape((-1, 1))
        kernel -= K_pred_cols
        kernel += self.__K_mean

        return kernel

    def transform(self, X):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Predicting data

        Returns
        -------
        X : shape (n_samples, n_components)
            The data of dimensionality reduction
        '''
        kernel = self.__kernel_func(self.__X, X, self.__sigma)
        kernel = self.__kernel_centeralization(kernel)
        pc = kernel.dot(self.__eig_vectors / np.sqrt(self.__eig_values))

        if self.__n_components == 2 and self.__visualize:
            metrics.scatter_feature(pc)

        return pc

class ZCAWhiten:
    def __init__(self, method=''):
        '''
        Parameters
        ----------
        method : SVD or not
        '''
        self.__method = method
    
    def fit_transform(self, X):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data

        Returns
        -------
        X : shape (n_samples, n_components)
            The data whitened
        '''
        n_samples = X.shape[0]

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

        self.__std = np.sqrt(self.__eig_values.reshape((1, -1)) / (n_samples - 1))

        return self.transform(X)

    def transform(self, X):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Predicting data

        Returns
        -------
        X : shape (n_samples, n_components)
            The data whitened
        '''
        X_sub_mean = X - self.__mean

        pc = X_sub_mean.dot(self.__eig_vectors)
        pc /= self.__std

        return pc.dot(self.__eig_vectors.T)