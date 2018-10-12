import numpy as np

class knn:
    def fit(self, X, y, k):
        self.__X = X
        self.__y = y
        self.__k = k

    def predict(self, X):
        data_number = X.shape[0]

        y_pred = np.zeros((data_number, 1))
        for i in range(data_number):
            distance = np.linalg.norm(X[i] - self.__X, axis=1)
            sort = np.argsort(distance)

            y_pred[i] = np.argmax(np.bincount(self.__y[sort[:self.__k]].flatten().astype(int)))
        
        return y_pred