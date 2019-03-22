import numpy as np
import cvxopt
import distance

class SVDD:
    @property
    def center(self):
        return self.__center

    @property
    def radius(self):
        return self.__radius

    def __qp(self, X, kernel, C):       
        n_samples = X.shape[0]

        P = 2 * kernel

        q = -kernel[range(n_samples), range(n_samples)].reshape(-1, 1)

        G = np.vstack((-np.eye(n_samples), np.eye(n_samples)))

        h = np.hstack((np.zeros(n_samples), np.full(n_samples, C)))

        A = np.full((1, n_samples), 1.0)

        b = np.ones(1)

        res = cvxopt.solvers.qp(cvxopt.matrix(P), cvxopt.matrix(q), cvxopt.matrix(G), cvxopt.matrix(h), cvxopt.matrix(A), cvxopt.matrix(b))
        alpha = np.array(res['x']).ravel()

        support_items = np.flatnonzero(np.isclose(alpha, 0) == False)
        self.__X_support = X[support_items]
        self.__a_support = alpha[support_items]

        free_items = np.flatnonzero(self.__a_support < C)
        X_free = self.__X_support[free_items]
        
        self.__center = self.__a_support.dot(self.__X_support)
        self.__radius = np.mean(distance.euclidean_distance(self.__center, X_free))

    def fit(self, X, kernel_func, C, sigma=1):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data
        kernel_func : kernel algorithm see also kernel.py
        C : Penalty parameter C of the error term
        sigma : Parameter for rbf kernel
        '''
        self.__sigma = sigma
        self.__kernel_func = kernel_func

        kernel = self.__kernel_func(X, X, self.__sigma)
        self.__qp(X, kernel, C)
        
    def predict(self, X):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Predicting data

        Returns
        -------
        y : shape (n_samples,)
            whether anormal per sample, True or False
        '''
        return self.__score(X) <= self.__radius

    def __score(self, X):
        n_samples = X.shape[0]

        scores = np.zeros(n_samples)
        for i in range(n_samples):
            x = X[i].reshape((1, -1))
            kernel1 = self.__kernel_func(x, x, self.__sigma)
            kernel2 = self.__kernel_func(x, self.__X_support, self.__sigma)
            kernel3 = self.__kernel_func(self.__X_support, self.__X_support, self.__sigma)
            scores[i] = kernel1 - 2 * self.__a_support.dot(kernel2) + self.__a_support.dot(kernel3).dot(self.__a_support.T)

        return np.sqrt(scores)