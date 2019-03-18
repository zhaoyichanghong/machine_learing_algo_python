import numpy as np

class KNN:
    def fit(self, X, y, n_neighbors, distance):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data
        y : shape (n_samples, n_classes)
            Target values
        n_neighbors : Number of neighbors
        distance : Distance algorithm, see also distance.py
        '''
        self.__X = X
        self.__y = y
        self.__n_neighbors = n_neighbors
        self.__distance = distance

    def __predict(self, x):
        distances = self.__distance(x, self.__X)
        nearest_items = np.argpartition(distances, self.__n_neighbors - 1)[:self.__n_neighbors]
        return np.argmax(np.bincount(self.__y[nearest_items].astype(int)))

    def predict(self, X):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Predicting data

        Returns
        -------
        y : shape (n_samples,)
            Predicted class label per sample.
        '''
        return np.apply_along_axis(self.__predict, 1, X)