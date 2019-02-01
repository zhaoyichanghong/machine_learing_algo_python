import numpy as np

class KMeansPlus:
    @property
    def centers(self):
        return self.__centers

    def fit(self, X, cluster_number, epochs, distance):
        data_number, feature_number = X.shape
        data = X

        self.__distance = distance
        self.__centers = np.zeros((cluster_number, feature_number))

        index = np.random.choice(data_number, 1)
        self.__centers[0] = data[index]
        data = np.delete(data, index, 0)
        data_number -= 1

        for i in range(cluster_number - 1):
            centers_number = i + 1
            distances = np.zeros((data_number, centers_number))
            for j in range(centers_number):
                distances[:, j] = self.__distance(self.__centers[j], data)
            distances = np.argmin(distances, axis=1)

            p = distances ** 2 / np.sum(distances ** 2)
            index = np.random.choice(range(data_number), p=p.ravel())
            self.__centers[i+1] = data[index]
            data = np.delete(data, index, 0)
            data_number -= 1

        for _ in range(epochs):
            labels = self.predict(X)

            for i in range(cluster_number):
                self.__centers[i] = np.mean(X[np.flatnonzero(labels == i)], axis=0)

        return labels

    def predict(self, X):
        distances = np.apply_along_axis(self.__distance, 1, self.__centers, X).T
        return np.argmin(distances, axis=1)