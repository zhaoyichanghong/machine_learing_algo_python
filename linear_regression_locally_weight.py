import numpy as np

class linear_regression_locally_weight:
    def __locally_weight(self, x):
        return np.exp(np.sum((self.__X - x) ** 2, axis=1) / (-2 * (self.__k ** 2)))

    def fit(self, X, y, k):
        self.__X = np.insert(X, 0, 1, axis=1)
        self.__y = y
        self.__k = k

    def predict(self, X):
        data_number = X.shape[0]
        X = np.insert(X, 0, 1, axis=1)

        y_pred = []
        for i in range(data_number):
            weights = self.__locally_weight(X[i])
            Weights = np.diag(weights.flatten())

            w = np.linalg.pinv(self.__X.T.dot(Weights).dot(self.__X)).dot(self.__X.T).dot(Weights).dot(self.__y)
            y_pred.append(X[i].dot(w))

        return np.array(y_pred).reshape((-1, 1))