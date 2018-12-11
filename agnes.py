import numpy as np

class agnes:
    def fit_transform(self, X, cluster_number, distance):
        self.__distance = distance
        data_number, feature_number = X.shape

        clusters = []
        for i in range(data_number):
            clusters.append([X[i].reshape(-1, feature_number)])
        self.__centers = X

        for j in reversed(range(cluster_number, data_number)):
            distances = np.apply_along_axis(self.__distance, 1, self.__centers, self.__centers)
            near_indexes = divmod(np.argmin(distances + np.diag(np.full(j + 1, np.inf))), j + 1)

            clusters[near_indexes[0]] += clusters[near_indexes[1]]
            self.__centers[near_indexes[0]] = np.mean(clusters[near_indexes[0]], axis=0)
            
            del clusters[near_indexes[1]]
            self.__centers = np.delete(self.__centers, near_indexes[1], axis=0)
        
        return self.predict(X)

    def predict(self, X):
        distances = np.apply_along_axis(self.__distance, 1, self.__centers, X).T
        return np.argmin(distances, axis=1)