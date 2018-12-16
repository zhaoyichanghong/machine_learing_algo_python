import numpy as np

class KNN:
    def fit(self, X, y, k, distance):
        self.__X = X
        self.__y = y
        self.__k = k
        self.__distance = distance

    def __predict(self, x):
        distance = self.__distance(x, self.__X)
        nearest_items = np.argpartition(distance, self.__k - 1)[:self.__k][0]
        return np.argmax(np.bincount(self.__y[nearest_items].astype(int)))

    def predict(self, X):       
        return np.apply_along_axis(self.__predict, 1, X).reshape((-1, 1))