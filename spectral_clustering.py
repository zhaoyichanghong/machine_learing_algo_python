import numpy as np
import k_means
import kernel

class SpectralClustering:
    def fit(self, X, n_clusters, sigma):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data
        n_clusters : The number of clusters
        sigma : Parameter for gaussian kernel

        Returns
        -------
        y : shape (n_samples,)
            Predicted cluster label per sample.
        '''
        n_samples = X.shape[0]

        W = kernel.gaussian_kernel(X, X, sigma)
        W[range(n_samples), range(n_samples)] = 0

        D = np.diag(np.sum(W, axis=1))

        L = np.linalg.inv(D).dot(D - W)

        eig_values, eig_vectors = np.linalg.eigh(L)
        U = eig_vectors[:, :n_clusters]

        return k_means.KMeans().fit(U, n_clusters, 100)
