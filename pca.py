import numpy as np

def pca(X, component_number, whiten=False, method=''):
    data_number = data.shape[0]
    X = data - np.mean(data, axis=0)

    if method == 'svd':
        u, s, vh = np.linalg.svd(X)
        eig_values = s ** 2
        eig_vectors = vh.T
    else:
        conv = X.T.dot(X)
        eig_values, eig_vectors = np.linalg.eig(conv)
        sort = np.argsort(-eig_values)
        eig_values = eig_values[sort]
        eig_vectors = eig_vectors[:, sort]

    pc = X.dot(eig_vectors[:, :component_number])

    if whiten:
        pc /= np.sqrt(eig_values[:component_number].reshape((1, -1)) / (data_number - 1))

    return pc

def zca_whiten(data, method=''):
    data_number = data.shape[0]
    X = data - np.mean(data, axis=0)

    if method == 'svd':
        u, s, vh = np.linalg.svd(X)
        eig_values = s ** 2
        eig_vectors = vh.T
    else:
        conv = X.T.dot(X)
        eig_values, eig_vectors = np.linalg.eig(conv)

    pc = X.dot(eig_vectors)
    pc /= np.sqrt(eig_values.reshape((1, -1)) / (data_number - 1))

    return pc.dot(eig_vectors.T)