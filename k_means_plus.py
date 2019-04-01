import numpy as np
import distance

class KMeansPlus:
    @property
    def centers(self):
        return self.__centers

    def fit(self, X, n_clusters, epochs):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data
        n_clusters : The number of clusters
        epochs : The number of epochs

        Returns
        -------
        y : shape (n_samples,)
            Predicted cluster label per sample.
        '''
        n_features = X.shape[1]
        self.__centers = np.zeros((n_clusters, n_features))

        data = X
        for i in range(n_clusters):
            n_samples = data.shape[0]

            if i == 0:
                index = np.random.choice(n_samples, 1)
            else:
                p = distances ** 2 / np.sum(distances ** 2)
                index = np.random.choice(n_samples, p=p.ravel())

            self.__centers[i] = data[index]
            data = np.delete(data, index, 0)

            if i != n_clusters - 1:
                distances = np.min(np.apply_along_axis(distance.euclidean_distance, 1, self.__centers, data).T, axis=1)

        for _ in range(epochs):
            labels = self.predict(X)
            self.__centers = [np.mean(X[np.flatnonzero(labels == i)], axis=0) for i in range(n_clusters)]

        return self.predict(X)

    def predict(self, X):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Predicting data

        Returns
        -------
        y : shape (n_samples,)
            Predicted cluster label per sample.
        '''
        distances = np.apply_along_axis(distance.euclidean_distance, 1, self.__centers, X).T
        return np.argmin(distances, axis=1)