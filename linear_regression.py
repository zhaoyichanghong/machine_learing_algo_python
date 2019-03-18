import numpy as np
import matplotlib.pyplot as plt
import optimizer
import regularizer

class LinearRegressionGradientDescent:
    def __init__(self, debug=True):
        self.__debug = debug

    def fit(self, X, y, epochs, optimizer, regularizer=regularizer.Regularizer(0)):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data
        y : shape (n_samples,)
            Target values
        epochs : The number of epochs
        optimizer : Optimize algorithm, see also optimizer.py
        regularizer : Regularize algorithm, see also regularizer.py
        '''
        n_samples, n_features = X.shape

        self.__W = np.zeros(n_features)
        self.__b = 0

        if self.__debug:
            loss = []

        for _ in range(epochs):
            h = self.predict(X)

            g_W = X.T.dot(h - y) / n_samples + regularizer.regularize(self.__W)
            g_b = np.mean(h - y)
            g_W, g_b = optimizer.optimize([g_W, g_b])
            self.__W -= g_W
            self.__b -= g_b

            if self.__debug:
                h = self.predict(X)
                loss.append(np.mean((h - y) ** 2))

        if self.__debug:
            plt.plot(loss)
            plt.show()

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
        return X.dot(self.__W) + self.__b

class LinearRegressionNewton:
    def __init__(self, debug=True):
        self.__debug = debug

    def fit(self, X, y, epochs):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data
        y : shape (n_samples,)
            Target values
        epochs : The number of epochs
        '''
        n_samples, n_features = X.shape

        self.__W = np.zeros(n_features)
        self.__b = 0

        if self.__debug: 
            loss = []

        for _ in range(epochs):
            h = self.predict(X)

            g_W = X.T.dot(h - y)
            H_W = X.T.dot(X)
            self.__W -= np.linalg.pinv(H_W).dot(g_W)

            self.__b -= np.mean(h - y)

            if self.__debug:
                h = self.predict(X)
                loss.append(np.mean((h - y) ** 2))

        if self.__debug:
            plt.plot(loss)
            plt.show()

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
        return X.dot(self.__W) + self.__b

class LinearRegressionEquation:
    def fit(self, X, y):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data
        y : shape (n_samples,)
            Target values
        '''
        X_with_b = np.insert(X, 0, 1, axis=1)
        self.__W = np.linalg.pinv(X_with_b.T.dot(X_with_b)).dot(X_with_b.T).dot(y)

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
        X_with_b = np.insert(X, 0, 1, axis=1)
        return X_with_b.dot(self.__W)