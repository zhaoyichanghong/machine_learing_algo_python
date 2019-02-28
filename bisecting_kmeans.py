import numpy as np
import k_means
import distance

class BisectingKMeans:
    @property
    def centers(self):
        return self.__centers

    def fit(self, X, cluster_number):
        '''
        Parameters
        ----------
        X : shape (data_number, feature_number)
            Training data
        cluster_number : The number of clusters

        Returns
        -------
        y : shape (data_number, 1)
            Predicted cluster label per sample.
        '''
        data = X
        clusters = []
        while True:
            model = k_means.KMeans()
            label = model.fit(data, 2, 100)

            clusters.append(X[np.flatnonzero(label == 0)])
            clusters.append(X[np.flatnonzero(label == 1)])

            if len(clusters) == cluster_number:
                self.__centers = [np.mean(cluster, axis=0) for cluster in clusters]
                break

            sse = [np.var(cluster) for cluster in clusters]
            data = clusters[np.argmax(sse)]
            del clusters[np.argmax(sse)]

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