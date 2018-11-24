import numpy as np
import matplotlib.pyplot as plt
import metrics
import regularizer

class logistic_regression_gradient_descent:
    def __init__(self, debug=True):
        self.__debug = debug

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y, epochs, optimizer, regularizer=regularizer.regularizer(0)):
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
                y_hat = self.score(X)
                loss.append(np.mean(-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)))
                accuracy.append(metrics.accuracy(y, np.around(y_hat)))

        if self.__debug:
            _, ax_loss = plt.subplots()
            ax_loss.plot(loss, 'b')
            ax_accuracy = ax_loss.twinx()
            ax_accuracy.plot(accuracy, 'r')
            plt.show()

    def predict(self, X):
        return np.around(self.score(X))

    def score(self, X):
        return self.__sigmoid(X.dot(self.__W) + self.__b)

class logistic_regression_newton:
    def __init__(self, debug=True):
        self.__debug = debug

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y, epochs):
        data_number, feature_number = X.shape

        self.__W = np.zeros((feature_number, 1))
        self.__b = 0

        if self.__debug:
            accuracy = []
            loss = []

        for _ in range(epochs):
            h = self.score(X)

            w_g = X.T.dot(h - y)
            A = np.diag((h * (1 - h)).flatten())
            w_H = X.T.dot(A).dot(X)
            self.__W -= np.linalg.pinv(w_H).dot(w_g)
            
            b_g = np.sum(h - y)
            b_H = np.sum(h * (1 - h))
            self.__b -= b_g / b_H
            
            if self.__debug:
                y_hat = self.score(X)
                loss.append(np.mean(-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)))
                accuracy.append(metrics.accuracy(y, np.around(y_hat)))

        if self.__debug:
            _, ax_loss = plt.subplots()
            ax_loss.plot(loss, 'b')
            ax_accuracy = ax_loss.twinx()
            ax_accuracy.plot(accuracy, 'r')
            plt.show()

    def predict(self, X):
        return np.around(self.score(X))

    def score(self, X):
        return self.__sigmoid(X.dot(self.__W) + self.__b)