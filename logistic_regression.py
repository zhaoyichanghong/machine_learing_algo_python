import numpy as np
import matplotlib.pyplot as plt
import metrics
import regularizer
import scipy

class LogisticRegressionGradientDescent:
    def __init__(self, debug=True):
        self.__debug = debug

    def fit(self, X, y, epochs, optimizer, regularizer=regularizer.Regularizer(0)):
        '''
        Parameters
        ----------
        X : shape (data_number, feature_number)
            Training data
        y : shape (data_number, 1)
            Target values
        epochs : The number of epochs
        optimizer : Optimize algorithm, see also optimizer.py
        regularizer : Regularize algorithm, see also regularizer.py
        '''
        data_number, feature_number = X.shape

        self.__W = np.zeros((feature_number, 1))
        self.__b = 0

        if self.__debug:
            accuracy = []
            loss = []

        for _ in range(epochs):
            h = self.score(X)

            g_W = X.T.dot(h - y) / data_number + regularizer.regularize(self.__W)
            g_b = np.mean(h - y)
            g_W, g_b = optimizer.optimize([g_W, g_b])
            self.__W -= g_W
            self.__b -= g_b

            if self.__debug:
                h = self.score(X)
                loss.append(np.mean(-y * np.log(h) - (1 - y) * np.log(1 - h)))
                accuracy.append(metrics.accuracy(y, np.around(h)))

        if self.__debug:
            _, ax_loss = plt.subplots()
            ax_loss.plot(loss, 'b')
            ax_accuracy = ax_loss.twinx()
            ax_accuracy.plot(accuracy, 'r')
            plt.show()

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
        return np.around(self.score(X))

    def score(self, X):
        '''
        Parameters
        ----------
        X : shape (data_number, feature_number)
            Predicting data

        Returns
        -------
        y : shape (data_number, 1)
            Predicted score per sample.
        '''
        return scipy.special.expit(X.dot(self.__W) + self.__b)

class LogisticRegressionNewton:
    def __init__(self, debug=True):
        self.__debug = debug

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

        if self.__debug:
            accuracy = []
            loss = []

        for _ in range(epochs):
            h = self.score(X)

            g_W = X.T.dot(h - y)
            A = np.diag((h * (1 - h)).ravel())
            H_W = X.T.dot(A).dot(X)
            self.__W -= np.linalg.pinv(H_W).dot(g_W)
            
            g_b = np.sum(h - y)
            H_b = np.sum(h * (1 - h))
            self.__b -= g_b / H_b
            
            if self.__debug:
                h = self.score(X)
                loss.append(np.mean(-y * np.log(h) - (1 - y) * np.log(1 - h)))
                accuracy.append(metrics.accuracy(y, np.around(h)))

        if self.__debug:
            _, ax_loss = plt.subplots()
            ax_loss.plot(loss, 'b')
            ax_accuracy = ax_loss.twinx()
            ax_accuracy.plot(accuracy, 'r')
            plt.show()

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
        return np.around(self.score(X))

    def score(self, X):
        '''
        Parameters
        ----------
        X : shape (data_number, feature_number)
            Predicting data

        Returns
        -------
        y : shape (data_number, 1)
            Predicted score per sample.
        '''
        return scipy.special.expit(X.dot(self.__W) + self.__b)