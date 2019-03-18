import numpy as np
import distance

class Agnes:
    def fit(self, X, n_clusters):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data
        n_clusters : The number of clusters

        Returns
        -------
        y : shape (n_samples,)
            Predicted cluster label per sample.
        '''
        n_samples = X.shape[0]

        clusters = [[i] for i in range(n_samples)]
        for j in reversed(range(n_clusters, n_samples)):
            centers = np.array([np.mean(X[cluster], axis=0).ravel() for cluster in clusters])
            distances = np.apply_along_axis(distance.euclidean_distance, 1, centers, centers)
            near_indexes = np.unravel_index(np.argmin(distances + np.diag(np.full(j + 1, np.inf))), distances.shape)

            clusters[near_indexes[0]].extend(clusters[near_indexes[1]])
            
            del clusters[near_indexes[1]]
        
        y = np.zeros(n_samples)
        for i in range(len(clusters)):
            y[clusters[i]] = i

        return y