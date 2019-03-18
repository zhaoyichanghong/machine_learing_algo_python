import numpy as np
from scipy.spatial.distance import pdist, squareform

class MultidimensionalScaling:
    def fit_transform(self, X, n_components):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data
        n_components : Number of components to keep

        Returns
        -------
        X : shape (n_samples, n_components)
            The data of dimensionality reduction
        '''
        n_samples = X.shape[0]

        D = squareform(pdist(X))

        J = np.eye(n_samples) - np.full_like(D, 1 / n_samples)
        B = -J.dot(D).dot(J) / 2

        eig_values, eig_vectors = np.linalg.eigh(B)
        eig_values = eig_values[::-1][:n_components]
        eig_vectors = eig_vectors[:, ::-1][:, :n_components]

        return eig_vectors * np.sqrt(eig_values)