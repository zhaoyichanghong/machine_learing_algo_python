import numpy as np
import distance

class KMeans:
    @property
    def centers(self):
        return self.__centers

    def fit(self, X, cluster_number, epochs):
        '''
        Parameters
        ----------
        X : shape (data_number, feature_number)
            Training data
        cluster_number : The number of clusters
        epochs : The number of epochs

        Returns
        -------
        y : shape (data_number, 1)
            Predicted cluster label per sample.
        '''
        data_number = X.shape[0]
        self.__centers = X[np.random.choice(data_number, cluster_number)]

        for _ in range(epochs):
            labels = self.predict(X)
            self.__centers = [np.mean(X[np.flatnonzero(labels == i)], axis=0) for i in range(cluster_number)]

        return self.predict(X)

    def predict(self, X):
        '''
        Parameters
        ----------
        X : shape (data_number, feature_number)
            Predicting data

        Returns
        -------
        y : shape (data_number, 1)
            Predicted cluster label per sample.
        '''
        distances = np.apply_along_axis(distance.euclidean_distance, 1, self.__centers, X).T
        return np.argmin(distances, axis=1)