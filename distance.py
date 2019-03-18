import numpy as np

def euclidean_distance(center, X):
    '''
    Parameters
    ----------
    X : shape (n_samples, n_features)
        The other end points of distance
    center : shape (n_features,)
             The one end point of distance

    Returns
    -------
    distance : shape (n_samples, 1)
               Euclidean distances to center
    '''
    return np.linalg.norm(X - center, axis=1)

def manhattan_distance(center, X):
    '''
    Parameters
    ----------
    X : shape (n_samples, n_features)
        The other end points of distance
    center : shape (n_features,)
             The one end point of distance

    Returns
    -------
    distance : shape (n_samples, 1)
               Manhattan distances to center
    '''
    return np.sum(np.abs(X - center), axis=1)

def chebyshev_distance(center, X):
    '''
    Parameters
    ----------
    X : shape (n_samples, n_features)
        The other end points of distance
    center : shape (n_features,)
             The one end point of distance

    Returns
    -------
    distance : shape (n_samples, 1)
               Chebyshev distances to center
    '''
    return np.max(np.abs(X - center), axis=1)

def mahalanobis_distance(center, X):
    '''
    Parameters
    ----------
    X : shape (n_samples, n_features)
        The other end points of distance
    center : shape (n_features,)
             The one end point of distance

    Returns
    -------
    distance : shape (n_samples, 1)
               Mahalanobis distances to center
    '''
    s_inv = np.linalg.inv(np.cov(X.T))
    distance = lambda x: np.sqrt((x - center).dot(s_inv).dot((x - center).T))
    return np.apply_along_axis(distance, 1, X)

def cosine_distance(center, X):
    '''
    Parameters
    ----------
    X : shape (n_samples, n_features)
        The other end points of distance
    center : shape (n_features,)
             The one end point of distance

    Returns
    -------
    distance : shape (n_samples, 1)
               Cosine distances to center
    '''
    return np.einsum('ij,ij->i', X, center.reshape((1, -1))) / (np.linalg.norm(X, axis=1) * np.linalg.norm(center))
