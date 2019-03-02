import numpy as np

def gaussian_kernel(X1, X2, sigma):
    '''
    Parameters
    ----------
    X1 : shape (data_number, feature_number)
    X2 : shape (data_number, feature_number)
    sigma : Parameter for gaussian kernel

    Returns
    -------
    kernel
    '''
    return np.exp((-(np.linalg.norm(X1[None, :, :] - X2[:, None, :], axis=2) ** 2)) / (2 * sigma ** 2))

def linear_kernel(X1, X2, *args):
    '''
    Parameters
    ----------
    X1 : shape (data_number, feature_number)
    X2 : shape (data_number, feature_number)
    *args : ignore

    Returns
    -------
    kernel
    '''
    return np.tensordot(X2, X1, axes=(1, 1))