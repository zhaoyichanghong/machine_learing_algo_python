import numpy as np
import matplotlib.pyplot as plt
import optimizer
import regularizer

class linear_regression_gradient_descent:
    def fit(self, X, y, epochs, optimizer, regularizer=regularizer.regularizer(0)):
        data_number, feature_number = X.shape

        self.__W = np.zeros((feature_number, 1))
        self.__b = 0

        loss = []
        for _ in range(epochs):
            h = X.dot(self.__W) + self.__b

            g_w = X.T.dot(h - y) / data_number + regularizer.regularize(self.__W)
            g_b = np.mean(h - y)
            g_w, g_b = optimizer.optimize(g_w, g_b)
            self.__W -= g_w
            self.__b -= g_b

            y_hat = X.dot(self.__W) + self.__b
            loss.append(np.mean((y_hat - y) ** 2))

        plt.plot(loss)
        plt.show()

    def predict(self, X):
        return X.dot(self.__W) + self.__b

class linear_regression_newton:
    def fit(self, X, y, epochs):
        data_number, feature_number = X.shape

        self.__W = np.zeros((feature_number, 1))
        self.__b = 0

        loss = []
        for _ in range(epochs):
            h = X.dot(self.__W) + self.__b

            w_g = X.T.dot(h - y)
            w_H = X.T.dot(X)
            self.__W -= np.linalg.pinv(w_H).dot(w_g)

            b_g = np.sum(h - y)
            b_H = data_number
            self.__b -= b_g / b_H

            y_hat = X.dot(self.__W) + self.__b
            loss.append(np.mean((y_hat - y) ** 2))

        plt.plot(loss)
        plt.show()

    def predict(self, X):
        return X.dot(self.__W) + self.__b

class linear_regression_equation:
    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.__W = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return X.dot(self.__W)