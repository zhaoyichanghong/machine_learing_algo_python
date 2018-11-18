import numpy as np

def gaussian_kernel(X1, X2, gamma):
    comput_kernel_for_row = lambda row: np.exp(-gamma * np.sum((X1 - row.reshape((1, -1))) ** 2, axis=1))
    return np.apply_along_axis(comput_kernel_for_row, 1, X2)

def linear_kernel(X1, X2, *args):
    return np.tensordot(X2, X1, axes=(1, 1))