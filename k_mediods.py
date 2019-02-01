import numpy as np

class KMediods:
    @property
    def centers(self):
        return self.__centers

    def fit(self, X, cluster_number, distance):
        data_number = X.shape[0]

        self.__distance = distance
        self.__centers = X[np.random.choice(data_number, cluster_number)]

        while True:
            labels = self.predict(X)

            centers = np.zeros_like(self.__centers)
            for i in range(cluster_number):
                cluster_data = X[np.flatnonzero(labels == i)]
                cluster_data_number = cluster_data.shape[0]
                errors = np.zeros(cluster_data_number)

                for j in range(cluster_data_number):
                    errors[j] = np.sum(self.__distance(cluster_data[j], cluster_data))
                
                centers[i] = cluster_data[np.argmin(errors)]
            
            if (centers == self.__centers).all():
                break
            else:
                self.__centers = centers

        return self.predict(X)

    def predict(self, X):
        distances = np.apply_along_axis(self.__distance, 1, self.__centers, X).T
        return np.argmin(distances, axis=1)