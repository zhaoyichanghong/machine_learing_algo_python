import numpy as np
import k_means

class BisectingKMeans:
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

        data = X
        clusters = []
        while True:
            model = k_means.KMeans()
            label = model.fit(data, 2, 100)

            clusters.append(np.flatnonzero(label == 0))
            clusters.append(np.flatnonzero(label == 1))

            if len(clusters) == cluster_number:
                break

            sse = [np.var(data[cluster]) for cluster in clusters]
            data = data[clusters[np.argmax(sse)]]
            del clusters[np.argmax(sse)]

        y = np.zeros(data_number)
        for i in range(len(clusters)):
            y[clusters[i]] = i

        return y