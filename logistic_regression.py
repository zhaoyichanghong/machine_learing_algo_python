import numpy as np
import matplotlib.pyplot as plt
import metrics

class logistic_regression:
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y, learning_rate, epochs):
        data_number, feature_number = X.shape

        self.__W = np.zeros((feature_number, 1))
        self.__b = 0

        accuracy = []
        loss = []
        for _ in range(epochs):
            h = self.__sigmoid(X.dot(self.__W) + self.__b)
            self.__W -= learning_rate * X.T.dot(h - y) / data_number
            self.__b -= learning_rate * np.mean(h - y)

            h = self.__sigmoid(X.dot(self.__W) + self.__b)
            loss.append(np.mean((h - y) ** 2))
            accuracy.append(metrics.accuracy(y, h > 0.5))

        _, ax_loss = plt.subplots()
        ax_loss.plot(loss, 'b')
        ax_accuracy = ax_loss.twinx()
        ax_accuracy.plot(accuracy, 'r')
        plt.show()

    def predict(self, X):
        return self.probability(X) > 0.5

    def probability(self, X):
        return self.__sigmoid(X.dot(self.__W) + self.__b)