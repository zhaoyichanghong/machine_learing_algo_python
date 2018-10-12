import numpy as np
import matplotlib.pyplot as plt
import metrics
import preprocess

class softmax_regression:
    def __softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)

    def fit(self, X, y, learning_rate, epochs):
        data_number, feature_number = X.shape

        encoder = preprocess.one_hot()
        y = encoder.fit_transform(y)
        class_number = y.shape[1]
        self.__classes = np.array(encoder.classes)

        self.__W = np.zeros((feature_number, class_number))
        self.__b = np.zeros((1, class_number))

        accuracy = []
        loss = []
        for _ in range(epochs):
            h = self.__softmax(X.dot(self.__W) + self.__b)
            self.__W -= learning_rate * X.T.dot(h - y) / data_number
            self.__b -= learning_rate * np.mean(h - y)

            y_hat = self.__softmax(X.dot(self.__W) + self.__b)
            loss.append(np.mean(-np.sum(y * np.log(y_hat), axis=1)))
            accuracy.append(metrics.accuracy(np.argmax(y, axis=1), np.argmax(y_hat, axis=1)))

        _, ax_loss = plt.subplots()
        ax_loss.plot(loss, 'b')
        ax_accuracy = ax_loss.twinx()
        ax_accuracy.plot(accuracy, 'r')
        plt.show()

    def predict(self, X):
        return self.__classes[np.argmax(self.probability(X), axis=1)]

    def probability(self, X):
        return self.__softmax(X.dot(self.__W) + self.__b)