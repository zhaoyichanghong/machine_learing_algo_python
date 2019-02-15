import numpy as np
import k_means
import distance
import kernel

class SpectralClustering:
    def fit(self, X, cluster_number, gamma):
        data_number = X.shape[0]

        W = kernel.gaussian_kernel(X, X, gamma)
        W[range(data_number), range(data_number)] = 0
        D = np.diag(np.sum(W, axis=1))
        L = np.linalg.inv(D).dot(D - W)
        eig_values, eig_vectors = np.linalg.eigh(L)
        U = eig_vectors[:, :cluster_number]

        model = k_means.KMeans()
        return model.fit(U, cluster_number, 100, distance.euclidean_distance)
