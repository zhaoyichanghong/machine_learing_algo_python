import numpy as np

class k_means:
    def __distance(self, x, means):
        return np.linalg.norm(x - means, axis=1)

    def fit_transform(self, X, cluster_number, epochs):
        data_number = X.shape[0]

        self.__means = X[np.random.choice(data_number, cluster_number)]

        clusters = []
        for _ in range(cluster_number):
            clusters.append([])

        for _ in range(epochs):
            for i in range(data_number):
                distances = self.__distance(X[i], self.__means)
                clusters[np.argmin(distances)].append(X[i])
            
            for i in range(cluster_number):
                self.__means[i] = np.mean(clusters[i], axis=0)

        return clusters

    def predict(self, x):
        distances = self.__distance(x, self.__means)
        return np.argmin(distances)