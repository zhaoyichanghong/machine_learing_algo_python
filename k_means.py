import numpy as np

class k_means:
    def fit_transform(self, X, cluster_number, epochs, distance):
        data_number = X.shape[0]

        self.__distance = distance
        self.__means = X[np.random.choice(data_number, cluster_number)]

        for _ in range(epochs):
            distances = np.apply_along_axis(self.__distance, 1, self.__means, X).T
            labels = np.argmin(distances, axis=1)

            for i in range(cluster_number):
                self.__means[i] = np.mean(X[np.where(labels == i)[0]], axis=0)

        return labels

    def predict(self, X):
        distances = np.apply_along_axis(self.__distance, 1, self.__means, X).T
        return np.argmin(distances, axis=1)