import numpy as np
import k_means
import kernel

class SpectralClustering:
    def fit(self, X, cluster_number, sigma):
        '''
        Parameters
        ----------
        X : shape (data_number, feature_number)
            Training data
        cluster_number : The number of clusters
        sigma : Parameter for gaussian kernel

        Returns
        -------
        y : shape (data_number,)
            Predicted cluster label per sample.
        '''
        data_number = X.shape[0]

        W = kernel.gaussian_kernel(X, X, sigma)
        W[range(data_number), range(data_number)] = 0

        D = np.diag(np.sum(W, axis=1))

        L = np.linalg.inv(D).dot(D - W)

        eig_values, eig_vectors = np.linalg.eigh(L)
        U = eig_vectors[:, :cluster_number]

        return k_means.KMeans().fit(U, cluster_number, 100)
