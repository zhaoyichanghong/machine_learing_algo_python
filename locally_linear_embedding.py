import numpy as np
import distance

class LLE:
    def fit_transform(self, X, neighber_number, component_number):
        data_number = X.shape[0]

        W = np.zeros((data_number, data_number))
        for i in range(data_number):
            distances = distance.euclidean_distance(X[i], X)
            distances[i] = np.inf
            nearest_item = np.argpartition(distances, neighber_number - 1)[:neighber_number]

            Z = (X[i] - X[nearest_item])
            Z = Z.dot(Z.T)
            Z_inv = np.linalg.pinv(Z)
            W[nearest_item, i] = np.sum(Z_inv, axis=1) / np.sum(Z_inv)
        
        M = (np.eye(data_number) - W).dot((np.eye(data_number) - W).T)
        eig_values, eig_vectors = np.linalg.eigh(M)
        
        return eig_vectors[:, 1:component_number+1]