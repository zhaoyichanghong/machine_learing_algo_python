import numpy as np
import kernel

class LinearRegressionLocallyWeight:
    def fit(self, X, y, sigma):
        '''
        Parameters
        ----------
        X : shape (data_number, feature_number)
            Training data
        y : shape (data_number, 1)
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
        X : shape (data_number, feature_number)
            Predicting data

        Returns
        -------
        y : shape (data_number, 1)
            Predicted value per sample.
        '''
        X = np.insert(X, 0, 1, axis=1)
        data_number, feature_number = X.shape

        W = np.zeros((data_number, feature_number))

        for i in range(data_number):
            Weights = np.diag(kernel.gaussian_kernel(self.__X, X[i].reshape(1, -1), self.__sigma).ravel())
            W[i] = (np.linalg.pinv(self.__X.T.dot(Weights).dot(self.__X)).dot(self.__X.T).dot(Weights).dot(self.__y)).ravel()

        return np.sum(X * W, axis=1, keepdims=True)