import numpy as np

class k_means:
    def fit_transform(self, X, cluster_number, epochs, distance):
        data_number = X.shape[0]

        self.__distance = distance
        self.__means = X[np.random.choice(data_number, cluster_number)]

        for _ in range(epochs):
            distances = np.zeros((data_number, cluster_number))
            for i in range(cluster_number):
                distances[:, i] = self.__distance(X, self.__means[i].reshape((1, -1)))

            labels = np.argmin(distances, axis=1)

            for i in range(cluster_number):
                self.__means[i] = np.mean(X[np.where(labels == i)[0]], axis=0)

        return labels

    def predict(self, X):
        data_number = X.shape[0]
        cluster_number = self.__means.shape[0]

        distances = np.zeros((data_number, cluster_number))
        for i in range(cluster_number):
            distances[:, i] = self.__distance(X, self.__means[i].reshape((1, -1)))

        return np.argmin(distances, axis=1)