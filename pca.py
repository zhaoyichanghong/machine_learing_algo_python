import numpy as np

def pca(X, component_number, method=''):
    X = X - np.mean(X, axis=0)

    if method == 'svd':
        u, s, vh = np.linalg.svd(X)
        eig_vectors = vh.T
    else:
        conv = X.T.dot(X)
        eig_values, eig_vectors = np.linalg.eig(conv)

    return X.dot(eig_vectors[:, :component_number])