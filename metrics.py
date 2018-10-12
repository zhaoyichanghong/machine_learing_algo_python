import numpy as np
import matplotlib.pyplot as plt

TP = lambda y_true, y_pred: np.sum(y_pred[np.where(y_true == 1)[0]] == 1)
FP = lambda y_true, y_pred: np.sum(y_pred[np.where(y_true == 0)[0]] == 1)
FN = lambda y_true, y_pred: np.sum(y_pred[np.where(y_true == 1)[0]] == 0)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    return TP(y_true, y_pred) / (TP(y_true, y_pred) + FP(y_true, y_pred))

def recall(y_true, y_pred):
    return TP(y_true, y_pred) / (TP(y_true, y_pred) + FN(y_true, y_pred))

def f_score(y_true, y_pred, beta):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (1 + beta ** 2) * p * r / (beta ** 2 * p + r)

def confusion_matrix(y_true, y_pred, labels=[]):
    class_number = len(labels)

    matrix = np.zeros((class_number, class_number))
    for r in range(class_number):
        for c in range(class_number):
            matrix[r, c] = np.sum(y_pred[np.where(y_true == r)[0]] == c)

    return matrix

def roc_curve(y_true, y_prob):
    TPR = []
    FPR = []
    for i in range(100):
        y_pred = y_prob > i / 100

        TPR.append(TP(y_true, y_pred) / np.sum(y_true == 1))
        FPR.append(FP(y_true, y_pred) / np.sum(y_true == 0))

    plt.plot(FPR, TPR)
    plt.show()

def r2_score(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)