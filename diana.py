import numpy as np

class diana:
    def fit_transform(self, X, cluster_number, distance):
        self.__distance = distance
        feature_number = X.shape[1]

        clusters = [X]
        while True:
            cluster_diameters = []
            for i in range(len(clusters)):
                distances = np.apply_along_axis(self.__distance, 1, clusters[i], clusters[i])
                cluster_diameters.append(np.max(distances))
            max_diameter_cluster = np.argmax(cluster_diameters)

            distances = np.apply_along_axis(self.__distance, 1, clusters[max_diameter_cluster], clusters[max_diameter_cluster])
            max_diff_index = np.argmax(np.mean(distances, axis=1))
            
            splinter_group = []
            old_party = clusters[max_diameter_cluster]
            splinter_group.append(clusters[max_diameter_cluster][max_diff_index])
            old_party = np.delete(old_party, max_diff_index, axis=0)

            for j in range(old_party.shape[0])[::-1]:
                distances_splinter = self.__distance(old_party[j], splinter_group)
                distances_old = self.__distance(old_party[j], np.delete(old_party, j, axis=0))
                if np.argmin(distances_splinter) < np.argmin(distances_old):
                    splinter_group.append(clusters[max_diameter_cluster][j])
                    old_party = np.delete(old_party, j, axis=0)

            del clusters[max_diameter_cluster]
            clusters.append(splinter_group)
            clusters.append(old_party)

            if len(clusters) == cluster_number:
                break

        self.__centers = np.zeros((cluster_number, feature_number))
        for i in range(cluster_number):
            self.__centers[i] = np.mean(clusters[i], axis=0)

        return self.predict(X)

    def predict(self, X):
        distances = np.apply_along_axis(self.__distance, 1, self.__centers, X).T
        return np.argmin(distances, axis=1)