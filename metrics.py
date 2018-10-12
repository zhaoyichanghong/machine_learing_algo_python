import numpy as np
import matplotlib.pyplot as plt

TP = lambda y_true, y_pred: np.sum(y_pred[np.where(y_true == 1)[0]] == 1)
TN = lambda y_true, y_pred: np.sum(y_pred[np.where(y_true == 0)[0]] == 0)
FP = lambda y_true, y_pred: np.sum(y_pred[np.where(y_true == 0)[0]] == 1)
FN = lambda y_true, y_pred: np.sum(y_pred[np.where(y_true == 1)[0]] == 0)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    if TP(y_true, y_pred) == 0:
        return 0

    return TP(y_true, y_pred) / (TP(y_true, y_pred) + FP(y_true, y_pred))

def recall(y_true, y_pred):
    if TP(y_true, y_pred) == 0:
        return 0

    return TP(y_true, y_pred) / (TP(y_true, y_pred) + FN(y_true, y_pred))

def f_score(y_true, y_pred, beta):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p * r == 0:
        return 0

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

    probs = np.sort(y_prob.flatten())
    for prob in probs:
        y_pred = y_prob > prob

        if TP(y_true, y_pred) == 0:
            TPR.append(0)
        else:
            TPR.append(TP(y_true, y_pred) / (TP(y_true, y_pred) + FN(y_true, y_pred)))

        if FP(y_true, y_pred) == 0:
            FPR.append(0)
        else:
            FPR.append(FP(y_true, y_pred) / (FP(y_true, y_pred) + TN(y_true, y_pred)))

    plt.plot(FPR, TPR)
    plt.show()

def auc(y_true, y_prob):
    rank = y_prob[:, 0].argsort()
    positives = np.where(y_true == 1)[0]
    rank_sum = np.sum([np.argwhere(rank == positive)[0][0] + 1 for positive in positives])

    positive_number = np.sum(y_true == 1)
    negative_number = np.sum(y_true == 0)
    
    return (rank_sum - positive_number * (positive_number + 1) / 2) / (positive_number * negative_number)

def r2_score(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)