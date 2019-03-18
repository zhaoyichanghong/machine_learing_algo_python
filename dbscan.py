import numpy as np

class Dbscan:
    def fit(self, X, radius, min_points, distance):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data
        radius : The maximum distance between two samples for them to be considered as in the same neighborhood
        min_points : The number of samples in a neighborhood for a point to be considered as a core point
        distance : Distance algorithm, see also distance.py

        Returns
        -------
        y : shape (n_samples,)
            Predicted cluster label per sample.
        '''
        n_samples = X.shape[0]
        kernel = set()
        data = set(range(n_samples))
        clusters = []
        distances = np.apply_along_axis(distance, 1, X, X)
        
        for i in range(n_samples):
            if np.sum(distances[i] < radius) >= min_points:
                kernel.add(i)

        while kernel:
            kernel_item = [list(kernel)[0]]
            cluster_kernel = set(kernel_item)
            cluster = set(kernel_item)
            data -= cluster

            while cluster_kernel:
                index = list(cluster_kernel)[0]
                neighbors = set(np.flatnonzero(distances[index] < radius))
                delta = neighbors & data
                cluster |= delta
                data -= delta
                cluster_kernel = cluster_kernel | (kernel & delta)
                cluster_kernel.remove(index)

            clusters.append(list(cluster))
            kernel -= cluster
        
        y = np.full(n_samples, -1)
        for i in range(len(clusters)):
            y[clusters[i]] = i

        return y