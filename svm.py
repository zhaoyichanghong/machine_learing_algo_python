import numpy as np
import cvxopt

class svm:
    def __gaussian_kernel(self, x1, x2):
        return np.exp(-self.__gamma * (x1 - x2).dot(x1 - x2))

    def __linear_kernel(self, x1, x2):
        return x1.dot(x2)

    def fit(self, X, y, kernel, C, gamma=1):
        self.__gamma = gamma
        self.__kernel = kernel

        data_number = X.shape[0]

        P = np.zeros((data_number, data_number))
        for i in range(data_number):
            for j in range(data_number):
                if self.__kernel == 'linear':
                    kernel = self.__linear_kernel(X[i], X[j])
                elif self.__kernel == 'rbf':
                    kernel = self.__gaussian_kernel(X[i], X[j])
                P[i, j] = y[i] * y[j] * kernel

        q = np.full((data_number, 1), -1.0)

        G = np.vstack((-np.eye(data_number), np.eye(data_number)))

        h = np.hstack((np.zeros(data_number), np.full(data_number, C)))

        A = y.T

        b = np.zeros(1)

        res = cvxopt.solvers.qp(cvxopt.matrix(P), cvxopt.matrix(q), cvxopt.matrix(G), cvxopt.matrix(h), cvxopt.matrix(A), cvxopt.matrix(b))
        alpha = np.array(res['x'])
        
        support_items = np.where(alpha > 1e-6)[0]
        support_items_number = len(support_items)
        self.__X_support = X[support_items]
        self.__y_support = y[support_items]
        self.__a_support = alpha[support_items]

        free_items = np.where(self.__a_support < (C - 1e-6))[0]
        X_free = X[free_items]
        y_free = y[free_items]

        kernel = np.zeros((support_items_number, 1))
        for i in range(support_items_number):
            if self.__kernel == 'linear':
                kernel[i] = self.__linear_kernel(self.__X_support[i], X_free[0])
            elif self.__kernel == 'rbf':
                kernel[i] = self.__gaussian_kernel(self.__X_support[i], X_free[0])

        self.__bias = y_free[0] - (self.__a_support * self.__y_support).T.dot(kernel)
        
    def predict(self, X):
        return np.sign(self.score(X))

    def score(self, X):
        data_number = X.shape[0]

        kernel = np.zeros((len(self.__X_support), data_number))
        for i in range(data_number):
            for j in range(len(self.__X_support)):
                if self.__kernel == 'linear':
                    kernel[j, i] = self.__linear_kernel(self.__X_support[j], X[i])
                elif self.__kernel == 'rbf':
                    kernel[j, i] = self.__gaussian_kernel(self.__X_support[j], X[i])

        return kernel.T.dot(self.__a_support * self.__y_support) + self.__bias