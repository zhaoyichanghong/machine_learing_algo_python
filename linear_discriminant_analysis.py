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
        X : shape (data_number, feature_number)
            Training data
        y : shape (data_number, 1)
            Data label

        Returns
        -------
        X : shape (data_number, n_components)
            The data of dimensionality reduction
        '''
        data_number, feature_number = X.shape
        classes = np.unique(y)
        class_number = len(classes)

        s_t = np.cov(X.T) * (data_number - 1)
        s_w = 0
        for i in range(class_number):
            items = np.flatnonzero(y == classes[i])
            number = len(items)           
            s_w += np.cov(X[items].T) * (number - 1)

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
        X : shape (data_number, feature_number)
            Predicting data

        Returns
        -------
        X : shape (data_number, n_components)
            The data of dimensionality reduction
        '''
        return X.dot(self.__eig_vectors)