import numpy as np
from scipy.stats import multivariate_normal

class GDA:
    def fit(self, X, y):
        '''
        Parameters
        ----------
        X : shape (data_number, feature_number)
            Training data
        y : shape (data_number, 1)
            Target labels
        '''
        data_number, feature_number = X.shape
        self.__classes = np.unique(y)
        class_number = len(self.__classes)
        
        self.__phi = np.zeros((class_number, 1))
        self.__means = np.zeros((class_number, feature_number))
        self.__sigma = 0
        for i in range(class_number):
            indexes = np.flatnonzero(y == self.__classes[i])

            self.__phi[i] = len(indexes) / data_number
            self.__means[i] = np.mean(X[indexes], axis=0)
            self.__sigma += np.cov(X[indexes].T) * (len(indexes) - 1)

        self.__sigma /= data_number

    def predict(self, X):
        '''
        Parameters
        ----------
        X : shape (data_number, feature_number)
            Predicting data

        Returns
        -------
        y : shape (data_number, 1)
            Predicted class label per sample.
        '''
        pdf = lambda mean: multivariate_normal.pdf(X, mean=mean, cov=self.__sigma)
        y_probs = np.apply_along_axis(pdf, 1, self.__means) * self.__phi

        return self.__classes[np.argmax(y_probs, axis=0)].reshape((-1, 1))