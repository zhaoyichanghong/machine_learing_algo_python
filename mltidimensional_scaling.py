import numpy as np
from scipy.spatial.distance import pdist, squareform

class MultidimensionalScaling:
    def fit_transform(self, X, n_components):
        '''
        Parameters
        ----------
        X : shape (data_number, feature_number)
            Training data
        n_components : Number of components to keep

        Returns
        -------
        X : shape (data_number, n_components)
            The data of dimensionality reduction
        '''
        data_number = X.shape[0]

        D = squareform(pdist(X))

        J = np.eye(data_number) - np.full_like(D, 1 / data_number)
        B = -J.dot(D).dot(J) / 2

        eig_values, eig_vectors = np.linalg.eigh(B)
        eig_values = eig_values[::-1][:n_components]
        eig_vectors = eig_vectors[:, ::-1][:, :n_components]

        return eig_vectors * np.sqrt(eig_values)