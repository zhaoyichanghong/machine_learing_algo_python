import numpy as np
import cvxopt

class svm:
    def __gaussian_kernel(self, X1, X2):
        comput_kernel_for_row = lambda row: np.exp(-self.__gamma * np.sum((X1 - row.reshape((1, -1))) ** 2, axis=1))
        return np.apply_along_axis(comput_kernel_for_row, 1, X2)

    def __linear_kernel(self, X1, X2):
        return np.tensordot(X1, X2, axes=(1, 1))

    def __qp(self, X, y, kernel, C, gamma):
        data_number = X.shape[0]

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

    def __smo(self, X, y, kernel, C, gamma, epochs):
        data_number = X.shape[0]

        alpha = np.zeros((data_number, 1))
        self.__bias = 0
        e = -y

        for _ in range(epochs):
            for i in range(data_number):
                hi = kernel[i].dot(alpha * y) + self.__bias
                if (y[i] * hi < 1 and alpha[i] < C) or (y[i] * hi > 1 and alpha[i] > 0):
                    j = np.argmax(np.abs(e - e[i]))

                    if y[i] == y[j]:
                        L = max(0, alpha[i] + alpha[j] - C)
                        H = min(C, alpha[i] + alpha[j])
                    else:
                        L = max(0, alpha[j] - alpha[i])
                        H = min(C, C + alpha[j] - alpha[i])
                    if L == H:
                        continue

                    eta = kernel[i, i] + kernel[j, j] - 2 * kernel[i, j]
                    if eta <= 0:
                        continue

                    alpha_j = alpha[j] + y[j] * (e[i] - e[j]) / eta

                    if alpha_j > H:
                        alpha_j = H
                    elif alpha_j < L:
                        alpha_j = L

                    alpha_i = alpha[i] + y[i] * y[j] * (alpha[j] - alpha_j)

                    bi = self.__bias - e[i] - y[i] * kernel[i, i] * (alpha_i - alpha[i]) - y[j] * kernel[i, j] * (alpha_j - alpha[j])
                    bj = self.__bias - e[j] - y[i] * kernel[i, j] * (alpha_i - alpha[i]) - y[j] * kernel[j, j] * (alpha_j - alpha[j])

                    if 0 < alpha_i and alpha_i < C:
                        self.__bias = bi
                    elif 0 < alpha_j and alpha_j < C:
                        self.__bias = bj
                    else:
                        self.__bias = (bi + bj) / 2

                    alpha[i] = alpha_i
                    alpha[j] = alpha_j

                    e[i] = kernel[i].dot(alpha * y) + self.__bias - y[i]
                    e[j] = kernel[j].dot(alpha * y) + self.__bias - y[j]
                
        support_items = np.where(alpha > 1e-6)[0]
        self.__X_support = X[support_items]
        self.__y_support = y[support_items]
        self.__a_support = alpha[support_items]

    def fit(self, X, y, kernel_option, C, gamma, solver, epochs=0):
        self.__gamma = gamma
        self.__kernel_option = kernel_option

        if self.__kernel_option == 'linear': 
            kernel = self.__linear_kernel(X, X)
        elif self.__kernel_option == 'rbf':
            kernel = self.__gaussian_kernel(X, X)
        
        if solver == 'qp':
            self.__qp(X, y, kernel, C, self.__gamma)
        elif solver == 'smo':
            self.__smo(X, y, kernel, C, self.__gamma, epochs)
        
    def predict(self, X):
        return np.sign(self.score(X))

    def score(self, X):
        if self.__kernel_option == 'linear':
            kernel = self.__linear_kernel(self.__X_support, X)
        elif self.__kernel_option == 'rbf':
            kernel = self.__gaussian_kernel(X, self.__X_support)

        return kernel.T.dot(self.__a_support * self.__y_support) + self.__bias