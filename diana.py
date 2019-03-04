import numpy as np
import distance

class Diana:
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
        data_number, feature_number = X.shape
        distances = np.apply_along_axis(distance.euclidean_distance, 1, X, X)

        clusters = [list(range(data_number))]
        while True:
            cluster_diameters = [np.max(distances[cluster][:, cluster]) for cluster in clusters]
            max_diameter_cluster = np.argmax(cluster_diameters)

            max_diff_index = np.argmax(np.mean(distances[clusters[max_diameter_cluster]][:, clusters[max_diameter_cluster]], axis=1))
            
            splinter_group = [clusters[max_diameter_cluster][max_diff_index]]
            old_party = clusters[max_diameter_cluster]
            del old_party[max_diff_index]
            while True:
                split = False
                for j in range(len(old_party))[::-1]:
                    distances_splinter = distances[old_party[j], splinter_group]
                    distances_old = distances[old_party[j], np.delete(old_party, j, axis=0)]
                    if np.mean(distances_splinter) <= np.mean(distances_old):
                        splinter_group.append(old_party[j])
                        del old_party[j]
                        split = True
                        break

                if split == False:
                    break

            del clusters[max_diameter_cluster]
            clusters.append(splinter_group)
            clusters.append(old_party)

            if len(clusters) == cluster_number:
                break

        y = np.zeros(data_number)
        for i in range(len(clusters)):
            y[clusters[i]] = i

        return y