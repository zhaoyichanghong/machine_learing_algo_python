import numpy as np
import metrics

class LDA:
    def __init__(self, n_components = None):
        '''
        Parameters
        ----------
        n_components : Number of components to keep
        '''
        self.__n_components = n_components

    def fit_transform(self, X, y):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data
        y : shape (n_samples,)
            Data label

        Returns
        -------
        X : shape (n_samples, n_components)
            The data of dimensionality reduction
        '''
        n_samples, n_features = X.shape
        classes = np.unique(y)
        n_classes = len(classes)

        s_t = np.cov(X.T) * (n_samples - 1)
        s_w = 0
        for i in range(n_classes):
            items = np.flatnonzero(y == classes[i])  
            s_w += np.cov(X[items].T) * (len(items)  - 1)

        s_b = s_t - s_w
        eig_values, eig_vectors = np.linalg.eigh(np.linalg.pinv(s_w).dot(s_b))
        self.__eig_vectors = eig_vectors[:, ::-1][:, :self.__n_components]

        pc = self.transform(X)
        
        if self.__n_components == 2:
            metrics.scatter_feature(pc, y)

        return pc

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
        return X.dot(self.__eig_vectors)