import numpy as np

class KMeans:
    def fit_transform(self, X, cluster_number, epochs, distance):
        data_number = X.shape[0]

        self.__distance = distance
        self.__centers = X[np.random.choice(data_number, cluster_number)]

        for _ in range(epochs):
            labels = self.predict(X)

            for i in range(cluster_number):
                self.__centers[i] = np.mean(X[np.flatnonzero(labels == i)], axis=0)

        return labels

    def predict(self, X):
        distances = np.apply_along_axis(self.__distance, 1, self.__centers, X).T
        return np.argmin(distances, axis=1)