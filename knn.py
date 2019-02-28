import numpy as np

class KNN:
    def fit(self, X, y, k, distance):
        '''
        Parameters
        ----------
        X : shape (data_number, feature_number)
            Training data
        y : shape (data_number, class_number)
            Target values
        k : Number of neighbors
        distance : Distance algorithm, see also distance.py
        '''
        self.__X = X
        self.__y = y
        self.__k = k
        self.__distance = distance

    def __predict(self, x):
        distances = self.__distance(x, self.__X)
        nearest_items = np.argpartition(distances, self.__k - 1)[:self.__k][0]
        return np.argmax(np.bincount(self.__y[nearest_items].astype(int)))

    def predict(self, X):
        '''
        Parameters
        ----------
        X : shape (data_number, feature_number)
            Predicting data

        Returns
        -------
        y : shape (data_number, 1)
            Predicted class label per sample.
        '''
        return np.apply_along_axis(self.__predict, 1, X).reshape((-1, 1))