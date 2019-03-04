import numpy as np
import distance

class Agnes:
    def fit(self, X, cluster_number):
        '''
        Parameters
        ----------
        X : shape (data_number, feature_number)
            Training data
        cluster_number : The number of clusters

        Returns
        -------
        y : shape (data_number,)
            Predicted cluster label per sample.
        '''
        data_number = X.shape[0]

        clusters = [[i] for i in range(data_number)]
        for j in reversed(range(cluster_number, data_number)):
            centers = np.array([np.mean(X[cluster], axis=0).ravel() for cluster in clusters])
            distances = np.apply_along_axis(distance.euclidean_distance, 1, centers, centers)
            near_indexes = np.unravel_index(np.argmin(distances + np.diag(np.full(j + 1, np.inf))), distances.shape)

            clusters[near_indexes[0]].extend(clusters[near_indexes[1]])
            
            del clusters[near_indexes[1]]
        
        y = np.zeros(data_number)
        for i in range(len(clusters)):
            y[clusters[i]] = i

        return y