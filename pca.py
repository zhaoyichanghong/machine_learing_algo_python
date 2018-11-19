import numpy as np
import metrics

def pca(data, component_number, whiten=False, method=''):
    data_number = data.shape[0]
    X = data - np.mean(data, axis=0)

    if method == 'svd':
        u, s, vh = np.linalg.svd(X)
        eig_values = s ** 2
        eig_vectors = vh.T
    else:
        conv = X.T.dot(X)
        eig_values, eig_vectors = np.linalg.eigh(conv)
        eig_values = eig_values[::-1]
        eig_vectors = eig_vectors[:, ::-1]

    pc = X.dot(eig_vectors[:, :component_number])

    if whiten:
        pc /= np.sqrt(eig_values[:component_number].reshape((1, -1)) / (data_number - 1))

    if component_number == 2:
        metrics.scatter_feature(pc)

    return pc

def kernel_pca(data, component_number, kernel_func, gamma=1, whiten=False):
    data_number = data.shape[0]
    X = data - np.mean(data, axis=0)

    I = np.full((data_number, data_number), 1 / data_number)
    K = kernel_func(X, X, gamma)
    K_hat = K - I.dot(K) - K.dot(I) + I.dot(K).dot(I)
    eig_values, eig_vectors = np.linalg.eigh(K_hat)
    eig_values = eig_values[::-1]
    eig_vectors = eig_vectors[:, ::-1]
    pc = eig_vectors[:, :component_number]

    if whiten:
        pc /= np.sqrt(eig_values[:component_number].reshape((1, -1)) / (data_number - 1))

    if component_number == 2:
        metrics.scatter_feature(pc)

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