import numpy as np

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    TP = np.sum(y_pred[np.where(y_true == 1)[0]] == 1)
    FP = np.sum(y_pred[np.where(y_true == 0)[0]] == 1)
    return TP / (TP + FP)

def recall(y_true, y_pred):
    TP = np.sum(y_pred[np.where(y_true == 1)[0]] == 1)
    FN = np.sum(y_pred[np.where(y_true == 1)[0]] == 0)
    return TP / (TP + FN)

def f_score(y_true, y_pred, beta):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (1 + beta ** 2) * p * r / (beta ** 2 * p + r)

def r2_score(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)