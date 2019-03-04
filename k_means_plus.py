import numpy as np
import distance

class KMeansPlus:
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
        y : shape (data_number,)
            Predicted cluster label per sample.
        '''
        feature_number = X.shape[1]
        self.__centers = np.zeros((cluster_number, feature_number))

        data = X
        for i in range(cluster_number):
            data_number = data.shape[0]

            if i == 0:
                index = np.random.choice(data_number, 1)
            else:
                p = distances ** 2 / np.sum(distances ** 2)
                index = np.random.choice(range(data_number), p=p.ravel())

            self.__centers[i] = data[index]
            data = np.delete(data, index, 0)

            if i != cluster_number - 1:
                distances = np.min(np.apply_along_axis(distance.euclidean_distance, 1, self.__centers, data).T, axis=1)

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
        y : shape (data_number,)
            Predicted cluster label per sample.
        '''
        distances = np.apply_along_axis(distance.euclidean_distance, 1, self.__centers, X).T
        return np.argmin(distances, axis=1)