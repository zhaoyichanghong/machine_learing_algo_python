import numpy as np
import metrics

class Pla:
    def fit(self, X, y):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data
        y : shape (n_samples,)
            Target values, 1 or -1
        '''
        n_samples, n_features = X.shape

        self.__W = np.zeros(n_features)
        self.__b = 0
        
        while True:
            for i in range(n_samples):
                h = self.predict(X[i])
                if y[i] * h <= 0:
                    self.__W += (y[i] * X[i]).reshape(self.__W.shape)
                    self.__b += y[i]

            h = self.predict(X)
            if metrics.accuracy(y, h) == 1:
                break

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
