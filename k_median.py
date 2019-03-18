import numpy as np
import distance

class KMedian:
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
        n_samples = X.shape[0]
        self.__centers = X[np.random.choice(n_samples, n_clusters)]

        for _ in range(epochs):
            labels = self.predict(X)
            self.__centers = [np.median(X[np.flatnonzero(labels == i)], axis=0) for i in range(n_clusters)]

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
        distances = np.apply_along_axis(distance.manhattan_distance, 1, self.__centers, X).T
        return np.argmin(distances, axis=1)