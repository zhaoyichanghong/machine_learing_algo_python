import numpy as np
from scipy.stats import multivariate_normal

class GDA:
    def fit(self, X, y):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data
        y : shape (n_samples,)
            Target labels
        '''
        n_samples, n_features = X.shape
        self.__classes = np.unique(y)
        n_classes = len(self.__classes)
        
        self.__phi = np.zeros((n_classes, 1))
        self.__means = np.zeros((n_classes, n_features))
        self.__sigma = 0
        for i in range(n_classes):
            indexes = np.flatnonzero(y == self.__classes[i])

            self.__phi[i] = len(indexes) / n_samples
            self.__means[i] = np.mean(X[indexes], axis=0)
            self.__sigma += np.cov(X[indexes].T) * (len(indexes) - 1)

        self.__sigma /= n_samples

    def predict(self, X):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Predicting data

        Returns
        -------
        y : shape (n_samples,)
            Predicted class label per sample.
        '''
        pdf = lambda mean: multivariate_normal.pdf(X, mean=mean, cov=self.__sigma)
        y_probs = np.apply_along_axis(pdf, 1, self.__means) * self.__phi

        return self.__classes[np.argmax(y_probs, axis=0)]