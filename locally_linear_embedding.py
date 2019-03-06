import numpy as np
from scipy.spatial.distance import pdist, squareform

class LLE:
    def fit_transform(self, X, n_neighbors, n_components):
        '''
        Parameters
        ----------
        X : shape (data_number, feature_number)
            Training data
        n_neighbors : number of neighbors to consider for each point
        n_components: number of coordinates for the manifold

        Returns
        -------
        X : shape (data_number, n_components)
            The data of dimensionality reduction
        '''
        data_number = X.shape[0]
        
        W = np.zeros((data_number, data_number))
        distances = squareform(pdist(X))
        distances[range(data_number), range(data_number)] = np.inf
        for i in range(data_number):
            nearest_item = np.argpartition(distances[i], n_neighbors)[:n_neighbors]

            Z = (X[i] - X[nearest_item]).dot((X[i] - X[nearest_item]).T)
            Z += np.eye(n_neighbors) * np.trace(Z) * 1e-8
            Z_inv = np.linalg.inv(Z)
            
            W[nearest_item, i] = np.sum(Z_inv, axis=1) / np.sum(Z_inv)
        
        M = (np.eye(data_number) - W).dot((np.eye(data_number) - W).T)
        eig_values, eig_vectors = np.linalg.eigh(M)
        
        return eig_vectors[:, 1:n_components+1]