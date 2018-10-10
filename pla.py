import numpy as np

class pla:
    def fit(self, X, y, learning_rate):
        data_number, feature_number = X.shape

        self.__W = np.zeros((feature_number, 1))
        self.__b = 0
        
        while True:
            for i in range(data_number):
                y_hat = X[i].dot(self.__W) + self.__b
                if y[i] * y_hat <= 0:
                    self.__W += learning_rate * y[i] * X[i].reshape(self.__W.shape)
                    self.__b += learning_rate * y[i]

            y_hat = X.dot(self.__W) + self.__b
            if all(y * y_hat > 0):
                break

    def predict(self, X):
        return np.sign(X.dot(self.__W) + self.__b)
