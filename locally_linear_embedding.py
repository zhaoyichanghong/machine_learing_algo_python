import numpy as np
from scipy.spatial.distance import pdist, squareform

class LLE:
    def fit_transform(self, X, n_neighbors, n_components):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data
        n_neighbors : number of neighbors to consider for each point
        n_components: number of coordinates for the manifold

        Returns
        -------
        X : shape (n_samples, n_components)
            The data of dimensionality reduction
        '''
        n_samples = X.shape[0]
        
        W = np.zeros((n_samples, n_samples))
        distances = squareform(pdist(X))
        distances[range(n_samples), range(n_samples)] = np.inf
        for i in range(n_samples):
            nearest_item = np.argpartition(distances[i], n_neighbors)[:n_neighbors]

            Z = (X[i] - X[nearest_item]).dot((X[i] - X[nearest_item]).T)
            Z += np.eye(n_neighbors) * np.trace(Z) * 1e-8
            Z_inv = np.linalg.inv(Z)
            
            W[nearest_item, i] = np.sum(Z_inv, axis=1) / np.sum(Z_inv)
        
        M = (np.eye(n_samples) - W).dot((np.eye(n_samples) - W).T)
        eig_values, eig_vectors = np.linalg.eigh(M)
        
        return eig_vectors[:, 1:n_components+1]