import numpy as np
import matplotlib.pyplot as plt

class linear_regression_gradient_descent:
    def fit(self, X, y, learning_rate, epochs):
        data_number, feature_number = X.shape

        self.__W = np.zeros((feature_number, 1))
        self.__b = 0

        loss = []
        for _ in range(epochs):
            h = X.dot(self.__W) + self.__b
            self.__W -= learning_rate * X.T.dot(h - y) / data_number
            self.__b -= learning_rate * np.mean(h - y)

            h = X.dot(self.__W) + self.__b
            loss.append(np.mean((h - y) ** 2))

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