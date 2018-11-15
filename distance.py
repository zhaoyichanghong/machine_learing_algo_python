import numpy as np

def euclidean_distance(center, X):
    center.reshape((1, -1))
    return np.linalg.norm(X - center, axis=1)

def manhattan_distance(center, X):
    center.reshape((1, -1))
    return np.sum(np.abs(X - center), axis=1)

def chebyshev_distance(center, X):
    center.reshape((1, -1))
    return np.max(np.abs(X - center), axis=1)

def mahalanobis_distance(center, X):
    s_inv = np.linalg.inv(np.cov(X.T))
    distance = lambda x: np.sqrt((x - center).dot(s_inv).dot((x - center).T))
    return np.apply_along_axis(distance, 1, X)

def cosine_distance(center, X):
    center.reshape((1, -1))
    return np.einsum('ij,ij->i', X, center) / (np.linalg.norm(X, axis=1) * np.linalg.norm(center))
