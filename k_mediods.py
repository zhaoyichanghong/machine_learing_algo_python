import numpy as np

class KMediods:
    def fit(self, X, cluster_number, distance):
        '''
        Parameters
        ----------
        X : shape (data_number, feature_number)
            Training data
        cluster_number : The number of clusters
        distance : Distance algorithm, see also distance.py

        Returns
        -------
        y : shape (data_number,)
            Predicted cluster label per sample.
        '''
        data_number = X.shape[0]
        self.__distance = distance
        distances = np.apply_along_axis(self.__distance, 1, X, X)
        
        centers = np.random.choice(data_number, cluster_number)
        while True:
            y = np.argmin(distances[centers], axis=0)

            centers_tmp = np.zeros_like(centers)
            for i in range(cluster_number):
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
        X : shape (data_number, feature_number)
            Predicting data

        Returns
        -------
        y : shape (data_number,)
            Predicted cluster label per sample.
        '''
        distances = np.apply_along_axis(self.__distance, 1, self.__centers, X).T
        return np.argmin(distances, axis=1)