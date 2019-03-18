import numpy as np
import k_means

class BisectingKMeans:
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

        data = X
        clusters = []
        while True:
            model = k_means.KMeans()
            label = model.fit(data, 2, 100)

            clusters.append(np.flatnonzero(label == 0))
            clusters.append(np.flatnonzero(label == 1))

            if len(clusters) == n_clusters:
                break

            sse = [np.var(data[cluster]) for cluster in clusters]
            data = data[clusters[np.argmax(sse)]]
            del clusters[np.argmax(sse)]

        y = np.zeros(n_samples)
        for i in range(len(clusters)):
            y[clusters[i]] = i

        return y