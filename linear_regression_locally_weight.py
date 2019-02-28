import numpy as np

class LinearRegressionLocallyWeight:
    def __locally_weight(self, x):
        return np.exp(np.sum((self.__X - x) ** 2, axis=1) / (-2 * (self.__gamma ** 2)))

    def fit(self, X, y, gamma):
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
        self.__gamma = gamma

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
            Weights = np.diag(self.__locally_weight(X[i]).ravel())

            W[i] = (np.linalg.pinv(self.__X.T.dot(Weights).dot(self.__X)).dot(self.__X.T).dot(Weights).dot(self.__y)).ravel()

        return np.sum(X * W, axis=1, keepdims=True)