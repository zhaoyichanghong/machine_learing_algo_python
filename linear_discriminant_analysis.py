import numpy as np
import metrics

class lda:
    def __init__(self, component_number = None):
        self.__component_number = component_number

    def fit_transform(self, X, y):
        data_number, feature_number = X.shape
        classes = np.unique(y)
        class_number = len(classes)

        s_t = np.cov(X.T) * (data_number - 1)
        s_w = 0
        for i in range(class_number):
            items = np.where(y == classes[i])[0]
            number = len(items)           
            s_w += np.cov(X[items].T) * (number - 1)

        s_b = s_t - s_w
        eig_values, eig_vectors = np.linalg.eigh(np.linalg.pinv(s_w).dot(s_b))
        self.__eig_vectors = eig_vectors[:, ::-1][:, :self.__component_number]

        pc = X.dot(self.__eig_vectors)
        if self.__component_number == 2:
            metrics.scatter_feature(pc, y)

        return pc

    def transform(self, X):
        return X.dot(self.__eig_vectors)