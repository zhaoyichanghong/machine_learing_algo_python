import numpy as np
import k_means

class BisectingKMeans:
    @property
    def centers(self):
        return self.__centers

    def fit(self, X, cluster_number, distance):
        feature_number = X.shape[1]

        self.__distance = distance
        data = X
        clusters = []

        while True:
            model = k_means.KMeans()
            label = model.fit(data, 2, 100, self.__distance)

            clusters.append(X[np.flatnonzero(label == 0)])
            clusters.append(X[np.flatnonzero(label == 1)])

            if len(clusters) == cluster_number:
                self.__centers = np.zeros((cluster_number, feature_number))
                for j in range(len(clusters)):
                    self.__centers[j] = np.mean(clusters[j], axis=0)
                break

            sse = np.zeros(len(clusters))
            for j in range(len(clusters)):
                sse[j] = np.var(clusters[j])
            data = clusters[np.argmax(sse)]
            del clusters[np.argmax(sse)]

        return self.predict(X)

    def predict(self, X):
        distances = np.apply_along_axis(self.__distance, 1, self.__centers, X).T
        return np.argmin(distances, axis=1)