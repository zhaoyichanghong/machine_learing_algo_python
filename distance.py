import numpy as np

def euclidean_distance(X, center):
    return np.linalg.norm(X - center, axis=1)

def manhattan_distance(X, center):
    return np.sum(np.abs(X - center), axis=1)

def chebyshev_distance(X, center):
    return np.max(np.abs(X - center), axis=1)

def mahalanobis_distance(X, center):
    data_number = X.shape[0]

    s_inv = np.linalg.inv(np.cov(X.T))

    distance = np.zeros(data_number)
    for i in range(data_number):
        distance[i] = np.sqrt((X[i] - center).dot(s_inv).dot((X[i] - center).T))
        
    return distance

def cosine_distance(X, center):
    return np.einsum('ij,ij->i', X, center) / (np.linalg.norm(X, axis=1) * np.linalg.norm(center))
