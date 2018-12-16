import numpy as np
import matplotlib.pyplot as plt
import metrics
import regularizer

class SoftmaxRegression:
    def __init__(self, debug=True):
        self.__debug = debug

    def __softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def fit(self, X, y, epochs, optimizer, regularizer=regularizer.Regularizer(0)):
        data_number, feature_number = X.shape
        class_number = y.shape[1]

        self.__W = np.zeros((feature_number, class_number))
        self.__b = np.zeros((1, class_number))

        if self.__debug:
            accuracy = []
            loss = []
            
        for _ in range(epochs):
            h = self.score(X)

            g_W = X.T.dot(h - y) / data_number + regularizer.regularize(self.__W)
            g_b = np.mean(h - y, axis=0)
            g_W, g_b = optimizer.optimize([g_W, g_b])
            self.__W -= g_W
            self.__b -= g_b

            if self.__debug:
                h = self.score(X)
                loss.append(np.mean(-np.sum(y * np.log(h), axis=1)))
                accuracy.append(metrics.accuracy(np.argmax(y, axis=1), np.argmax(h, axis=1)))

        if self.__debug:
            _, ax_loss = plt.subplots()
            ax_loss.plot(loss, 'b')
            ax_accuracy = ax_loss.twinx()
            ax_accuracy.plot(accuracy, 'r')
            plt.show()

    def predict(self, X, classes):
        return classes[np.argmax(self.score(X), axis=1)].reshape((-1, 1))

    def score(self, X):
        return self.__softmax(X.dot(self.__W) + self.__b)