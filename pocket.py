import numpy as np
import metrics

class Pocket:
    def fit(self, X, y, learning_rate, epochs):
        feature_number = X.shape[1]

        self.__W = np.zeros((feature_number, 1))
        self.__b = 0

        accuracy = 0
        for _ in range(epochs):
            h = self.predict(X)
            error_index = np.random.choice(np.flatnonzero(y != h))

            W_tmp = self.__W + learning_rate * y[error_index] * X[error_index].reshape(self.__W.shape)
            b_tmp = self.__b + learning_rate * y[error_index]

            h = np.sign(X.dot(W_tmp) + b_tmp)
            accuracy_tmp = metrics.accuracy(y, h)
            if accuracy_tmp > accuracy:
                accuracy = accuracy_tmp
                self.__W = W_tmp
                self.__b = b_tmp

    def predict(self, X):
        return np.sign(X.dot(self.__W) + self.__b)