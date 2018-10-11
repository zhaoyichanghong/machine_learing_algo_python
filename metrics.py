import numpy as np

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def r2_score(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)