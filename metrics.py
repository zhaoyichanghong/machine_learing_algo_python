import numpy as np
import matplotlib.pyplot as plt

TP = lambda y_true, y_pred: np.sum(y_pred[np.flatnonzero(y_true == 1)] == 1)
TN = lambda y_true, y_pred: np.sum(y_pred[np.flatnonzero(y_true != 1)] != 1)
FP = lambda y_true, y_pred: np.sum(y_pred[np.flatnonzero(y_true != 1)] == 1)
FN = lambda y_true, y_pred: np.sum(y_pred[np.flatnonzero(y_true == 1)] != 1)

def accuracy(y_true, y_pred):
    '''
    Parameters
    ----------
    y_true : shape (n_samples,)
             True label
    y_true : shape (n_samples,)
             Predicting label

    Returns
    -------
    accuracy
    '''
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    '''
    Parameters
    ----------
    y_true : shape (n_samples,)
             True label
    y_true : shape (n_samples,)
             Predicting label

    Returns
    -------
    precision
    '''
    if TP(y_true, y_pred) == 0:
        return 0

    return TP(y_true, y_pred) / (TP(y_true, y_pred) + FP(y_true, y_pred))

def recall(y_true, y_pred):
    '''
    Parameters
    ----------
    y_true : shape (n_samples,)
             True label
    y_true : shape (n_samples,)
             Predicting label

    Returns
    -------
    recall
    '''
    if TP(y_true, y_pred) == 0:
        return 0

    return TP(y_true, y_pred) / (TP(y_true, y_pred) + FN(y_true, y_pred))

def f_score(y_true, y_pred, beta=1):
    '''
    Parameters
    ----------
    y_true : shape (n_samples,)
             True label
    y_true : shape (n_samples,)
             Predicting label
    beta : Weight of precision in harmonic mean

    Returns
    -------
    f_score
    '''
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p * r == 0:
        return 0

    return (1 + beta ** 2) * p * r / (beta ** 2 * p + r)

def confusion_matrix(y_true, y_pred):
    '''
    Parameters
    ----------
    y_true : shape (n_samples,)
             True label
    y_true : shape (n_samples,)
             Predicting label

    Returns
    -------
    confusion matrix : shape (n_classes, n_classes)
    '''
    classes = np.unique(y_true)
    n_classes = len(classes)

    matrix = np.zeros((n_classes, n_classes))
    for r in range(n_classes):
        for c in range(n_classes):
            matrix[r, c] = np.sum(y_pred[np.flatnonzero(y_true == classes[r])] == classes[c])

    return matrix

def pr_curve(y_true, y_score):
    '''
    Parameters
    ----------
    y_true : shape (n_samples,)
             True label
    y_score : shape (n_samples,)
             Predicting score
    '''
    p = []
    r = []

    scores = np.sort(y_score.flatten())
    for score in scores:
        y_pred = y_score > score

        p.append(precision(y_true, y_pred))
        r.append(recall(y_true, y_pred))

    plt.plot(r, p)
    plt.show()

def roc_curve(y_true, y_score):
    '''
    Parameters
    ----------
    y_true : shape (n_samples,)
             True label
    y_score : shape (n_samples,)
             Predicting score
    '''
    TPR = []
    FPR = []

    scores = np.sort(y_score.flatten())
    for score in scores:
        y_pred = y_score > score

        TPR.append(recall(y_true, y_pred))

        if FP(y_true, y_pred) == 0:
            FPR.append(0)
        else:
            FPR.append(FP(y_true, y_pred) / (FP(y_true, y_pred) + TN(y_true, y_pred)))

    plt.plot(FPR, TPR)
    plt.show()

def auc(y_true, y_score):
    '''
    Parameters
    ----------
    y_true : shape (n_samples,)
             True label
    y_score : shape (n_samples,)
             Predicting score
    
    Returns
    -------
    auc
    '''
    n_samples = y_true.shape[0]

    rank = y_score.ravel().argsort()
    positives = np.flatnonzero(y_true == 1)
    rank_sum = np.sum([np.flatnonzero(rank == positive) + 1 for positive in positives])

    n_positive = len(positives)
    
    return (rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * (n_samples - n_positive))

def r2_score(y_true, y_pred):
    '''
    Parameters
    ----------
    y_true : shape (n_samples,)
             True value
    y_pred : shape (n_samples,)
             Predicting value
    
    Returns
    -------
    r2 score
    '''
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

def silhouette_coefficient(X, y, distance):
    '''
    Parameters
    ----------
    X : shape (n_samples, n_features)
        Training data
    y : shape (n_samples,)
        Target values
    distance : Distance algorithm, see also distance.py

    Returns
    -------
    silhouette coefficient
    '''
    n_samples = X.shape[0]

    s = []
    for i in range(n_samples):
        distances = distance(X[i], X)
        
        bs = []
        for cluster in np.unique(y):
            if y[i] == cluster:
                a = np.sum(distances[np.flatnonzero(y == cluster)]) / (np.sum(y == cluster) - 1 + 1e-8)
            else:
                bs.append(np.mean(distances[np.flatnonzero(y == cluster)]))
        b = np.min(bs)

        s.append((b - a) / max(a, b))
    
    return np.mean(s)

def scatter_feature(X, y=None):
    '''
    Parameters
    ----------
    X : shape (n_samples, n_features)
        Training data
    y : shape (n_samples,)
        Target values
    '''
    if y is None:
        plt.scatter(X[:, 0], X[:, 1])
    else:
        colors = ['r', 'b', 'g']
        labels = np.unique(y)
        for color, label in zip(colors, labels):
            class_data = X[np.flatnonzero(y == label)]
            plt.scatter(class_data[:, 0], class_data[:, 1], c=color)

    plt.show()

def learning_curve(train_X, train_y, train_ratios, test_X, test_y, fit, accuracy):
    '''
    Parameters
    ----------
    train_X : shape (n_samples, n_features)
              Training data
    train_y : shape (n_samples,)
              Target values
    train_ratios : Relative numbers of training examples that will be used to generate the learning curve
    test_X : shape (n_samples, n_features)
             Testing data
    test_y : shape (n_samples,)
             Target values
    fit : Fitting function
    accuracy : Accuracy function
    '''
    n_train_samples = train_X.shape[0]

    accuracy_train = []
    accuracy_test = []
    for i in (n_train_samples * train_ratios).astype(int):
        fit(train_X[:i], train_y[:i])
        accuracy_train.append(accuracy(train_X[:i], train_y[:i]))
        accuracy_test.append(accuracy(test_X, test_y))

    plt.plot(accuracy_train, 'r')
    plt.plot(accuracy_test, 'b')
    plt.show()

def information_value(X, y):
    '''
    Parameters
    ----------
    X : shape (n_samples, n_features)
              Training data
    y : shape (n_samples,)
              Target label

    Returns
    -------
    information values of each feature
    '''
    n_features = X.shape[1]

    n_positive_total = sum(y == 1)
    n_negative_total = sum(y == 0)

    ivs = []
    for i in range(n_features):
        iv = 0
        feature_labels = np.unique(X[:, i])
        for feature_label in feature_labels:
            indexes = np.flatnonzero(X[:, i] == feature_label)

            n_positive = sum(y[indexes] == 1)
            n_negative = sum(y[indexes] == 0)

            p_positive = n_positive / n_positive_total
            p_negative = n_negative / n_negative_total

            iv += (p_positive - p_negative) * np.log(p_positive + 1e-8 / p_negative + 1e-8)
        
        ivs.append(iv)

    return ivs

            

    
    