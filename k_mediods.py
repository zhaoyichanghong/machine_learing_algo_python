import numpy as np

class KMediods:
    def fit(self, X, n_clusters, distance):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data
        n_clusters : The number of clusters
        distance : Distance algorithm, see also distance.py

        Returns
        -------
        y : shape (n_samples,)
            Predicted cluster label per sample.
        '''
        n_samples = X.shape[0]
        self.__distance = distance
        distances = np.apply_along_axis(self.__distance, 1, X, X)
        
        centers = np.random.choice(n_samples, n_clusters)
        while True:
            y = np.argmin(distances[centers], axis=0)

            centers_tmp = np.zeros_like(centers)
            for i in range(n_clusters):
                indexes = np.flatnonzero(y == i)
                errors = np.sum(distances[indexes][:, indexes], axis=0)
                centers_tmp[i] = indexes[np.argmin(errors)]
            
            if (centers == centers_tmp).all():
                break
            else:
                centers = centers_tmp

        self.__centers = X[centers]
        return y

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
        distances = np.apply_along_axis(self.__distance, 1, self.__centers, X).T
        return np.argmin(distances, axis=1)