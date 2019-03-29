import numpy as np
import distance

class FCM:
    def fit(self, X, n_clusters, m, epochs):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data
        n_clusters : The number of clusters
        m : weighted index number
        epochs : The number of epochs

        Returns
        -------
        y : shape (n_samples,)
            Predicted cluster label per sample.
        '''
        n_samples, n_features = X.shape

        weights = np.zeros((n_samples, n_clusters))
        centers = X[np.random.choice(n_samples, n_clusters)]
        for _ in range(epochs):            
            distances = np.apply_along_axis(distance.euclidean_distance, 1, centers, X).T + 1e-8

            for i in range(n_clusters):
                weights[:, i] = 1 / np.sum((distances[:, i].reshape((-1, 1)) / distances) ** (2 / (m - 1)), axis=1)

            for i in range(n_clusters):
                centers[i] = np.sum(weights[:, i].reshape(-1, 1) ** m * X, axis=0) / np.sum(weights[:, i] ** m)

        return np.argmax(weights, axis=1)