import numpy as np

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)