import numpy as np
import kernel

class LinearRegressionLocallyWeight:
    def fit(self, X, y, sigma):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data
        y : shape (n_samples,)
            Target values
        gamma : For RBF kernel
        '''
        self.__X = np.insert(X, 0, 1, axis=1)
        self.__y = y
        self.__sigma = sigma

    def predict(self, X):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Predicting data

        Returns
        -------
        y : shape (n_samples,)
            Predicted value per sample.
        '''
        X = np.insert(X, 0, 1, axis=1)
        n_samples, n_features = X.shape

        W = np.zeros((n_samples, n_features))

        for i in range(n_samples):
            Weights = np.diag(kernel.gaussian_kernel(self.__X, X[i].reshape(1, -1), self.__sigma).ravel())
            W[i] = (np.linalg.pinv(self.__X.T.dot(Weights).dot(self.__X)).dot(self.__X.T).dot(Weights).dot(self.__y)).ravel()

        return np.sum(X * W, axis=1)