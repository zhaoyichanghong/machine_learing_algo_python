import numpy as np
import matplotlib.pyplot as plt
import metrics
import regularizer

class softmax_regression:
    def __init__(self, debug=True):
        self.__debug = debug

    def __softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)

    def fit(self, X, y, epochs, optimizer, regularizer=regularizer.regularizer(0)):
        data_number, feature_number = X.shape
        class_number = y.shape[1]

        self.__W = np.zeros((feature_number, class_number))
        self.__b = np.zeros((1, class_number))

        if self.__debug:
            accuracy = []
            loss = []
        for _ in range(epochs):
            h = self.__softmax(X.dot(self.__W) + self.__b)

            g_w = X.T.dot(h - y) / data_number + regularizer.regularize(self.__W)
            g_b = np.mean(h - y, axis=0)
            g_w, g_b = optimizer.optimize(g_w, g_b)
            self.__W -= g_w
            self.__b -= g_b

            if self.__debug:
                y_hat = self.__softmax(X.dot(self.__W) + self.__b)
                loss.append(np.mean(-np.sum(y * np.log(y_hat), axis=1)))
                accuracy.append(metrics.accuracy(np.argmax(y, axis=1), np.argmax(y_hat, axis=1)))

        if self.__debug:
            _, ax_loss = plt.subplots()
            ax_loss.plot(loss, 'b')
            ax_accuracy = ax_loss.twinx()
            ax_accuracy.plot(accuracy, 'r')
            plt.show()

    def predict(self, X, classes):
        return classes[np.argmax(self.probability(X), axis=1)]

    def probability(self, X):
        return self.__softmax(X.dot(self.__W) + self.__b)