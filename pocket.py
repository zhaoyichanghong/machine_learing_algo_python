import numpy as np
import metrics

class Pocket:
    def fit(self, X, y, epochs):
        '''
        Parameters
        ----------
        X : shape (data_number, feature_number)
            Training data
        y : shape (data_number, 1)
            Target values
        epochs : The number of epochs
        '''
        feature_number = X.shape[1]

        self.__W = np.zeros((feature_number, 1))
        self.__b = 0

        accuracy = 0
        for _ in range(epochs):
            h = self.predict(X)
            error_index = np.random.choice(np.flatnonzero(y != h))

            W_tmp = self.__W + (y[error_index] * X[error_index]).reshape(self.__W.shape)
            b_tmp = self.__b + y[error_index]

            h = np.sign(X.dot(W_tmp) + b_tmp)
            accuracy_tmp = metrics.accuracy(y, h)
            if accuracy_tmp > accuracy:
                accuracy = accuracy_tmp
                self.__W = W_tmp
                self.__b = b_tmp

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
        return np.sign(X.dot(self.__W) + self.__b)