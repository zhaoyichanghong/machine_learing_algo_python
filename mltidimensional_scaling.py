import numpy as np
import distance

class MultidimensionalScaling:
    def fit_transform(self, X, component_number):
        data_number = X.shape[0]

        D = np.zeros((data_number, data_number))
        for i in range(data_number):
            D[i] = distance.euclidean_distance(X[i], X)

        J = np.eye(data_number) - np.full_like(D, 1 / data_number)
        B = -J.dot(D).dot(J) / 2

        eig_values, eig_vectors = np.linalg.eigh(B)
        eig_values = eig_values[::-1][:component_number]
        eig_vectors = eig_vectors[:, ::-1][:, :component_number]
        
        return eig_vectors.dot(np.diag(np.sqrt(eig_values)))