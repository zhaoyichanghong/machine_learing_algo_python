import numpy as np
import metrics

class Pocket:
    def fit(self, X, y, epochs):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data
        y : shape (n_samples,)
            Target values, 1 or -1
        epochs : The number of epochs
        '''
        n_features = X.shape[1]

        self.__W = np.zeros(n_features)
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
        X : shape (n_samples, n_features)
            Predicting data

        Returns
        -------
        y : shape (n_samples,)
            Predicted class label per sample, 1 or -1
        '''
        return np.sign(X.dot(self.__W) + self.__b)