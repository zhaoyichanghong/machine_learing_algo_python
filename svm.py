import numpy as np
import cvxopt

class svm:
    def __gaussian_kernel(self, X1, x2):
        return np.exp(-self.__gamma * np.sum((X1 - x2) ** 2, axis=1))

    def __linear_kernel(self, X1, X2):
        return np.tensordot(X1, X2, axes=(1, 1))

    def fit(self, X, y, kernel_option, C, gamma=1):
        self.__gamma = gamma
        self.__kernel_option = kernel_option

        data_number = X.shape[0]

        if self.__kernel_option == 'linear': 
            kernel = self.__linear_kernel(X, X)
        elif self.__kernel_option == 'rbf':
            kernel = np.zeros((data_number, data_number))
            for i in range(data_number):
                kernel[i] = self.__gaussian_kernel(X, X[i])

        P = y.dot(y.T) * kernel

        q = np.full((data_number, 1), -1.0)

        G = np.vstack((-np.eye(data_number), np.eye(data_number)))

        h = np.hstack((np.zeros(data_number), np.full(data_number, C)))

        A = y.T

        b = np.zeros(1)

        res = cvxopt.solvers.qp(cvxopt.matrix(P), cvxopt.matrix(q), cvxopt.matrix(G), cvxopt.matrix(h), cvxopt.matrix(A), cvxopt.matrix(b))
        alpha = np.array(res['x'])

        support_items = np.where(alpha > 1e-6)[0]
        self.__X_support = X[support_items]
        self.__y_support = y[support_items]
        self.__a_support = alpha[support_items]

        free_items = np.where(self.__a_support < (C - 1e-6))[0]
        X_free = X[free_items]
        y_free = y[free_items]

        self.__bias = y_free[0] - (self.__a_support * self.__y_support).T.dot(kernel[support_items, free_items[0]])
        
    def predict(self, X):
        return np.sign(self.score(X))

    def score(self, X):
        data_number = X.shape[0]

        if self.__kernel_option == 'linear':
            kernel = self.__linear_kernel(self.__X_support, X)
        else:
            kernel = np.zeros((self.__X_support.shape[0], data_number))
            for i in range(data_number):
                kernel[:, i] = self.__gaussian_kernel(self.__X_support, X[i])

        return kernel.T.dot(self.__a_support * self.__y_support) + self.__bias