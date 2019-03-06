import numpy as np
import metrics
import matplotlib.pyplot as plt
import k_means_plus
import kernel

class RbfNet:
    def __init__(self, debug=True):
        self.__debug = debug

    def fit(self, X, y, units, epochs, optimizer):
        '''
        Parameters
        ----------
        X : shape (data_number, feature_number)
            Training data
        y : shape (data_number, 1)
            Target values, 1 or 0
        epochs : The number of epochs
        optimizer : Optimize algorithm, see also optimizer.py
        units : The number of unit
        '''
        data_number, feature_number = X.shape

        self.__units = units

        model = k_means_plus.KMeansPlus()
        model.fit(X, self.__units, 10)        
        self.__centers = model.centers
        
        self.__sigmas = np.ones((self.__units, 1))
        self.__weights = np.random.randn(self.__units, 1)
        
        if self.__debug:
            accuracy = []
            loss = []

        for _ in range(epochs):
            outs = np.zeros((data_number, self.__units))
            for i in range(self.__units):
                outs[:, i] = kernel.gaussian_kernel(X, self.__centers[i].reshape(1, -1), self.__sigmas[i])
            h = outs.dot(self.__weights)
            residual = h - y
            
            g_centers = np.zeros_like(self.__centers)
            g_sigmas = np.zeros_like(self.__sigmas)
            g_weights = np.zeros_like(self.__weights)
            for i in range(self.__units):
                g_centers[i] = self.__weights[i] * np.mean(residual * outs[:, i].reshape((-1, 1)) * (X - self.__centers[i]), axis=0) / (self.__sigmas[i] ** 2)
                g_sigmas[i] = self.__weights[i] * np.mean(residual * outs[:, i].reshape((-1, 1)) * (np.linalg.norm(X - self.__centers[i], axis=1).reshape((-1, 1)) ** 2), axis=0) / (self.__sigmas[i] ** 3)
                g_weights[i] = np.mean(residual * outs[:, i].reshape((-1, 1)), axis=0)
            
            g_centers, g_sigmas, g_weights = optimizer.optimize([g_centers, g_sigmas, g_weights])
            self.__centers -= g_centers
            self.__sigmas -= g_sigmas
            self.__weights -= g_weights

            if self.__debug:
                h = self.score(X)
                loss.append(np.mean((h - y) ** 2))
                accuracy.append(metrics.accuracy(y, np.around(h)))

        if self.__debug:
            _, ax_loss = plt.subplots()
            ax_loss.plot(loss, 'b')
            ax_accuracy = ax_loss.twinx()
            ax_accuracy.plot(accuracy, 'r')
            plt.show()

    def score(self, X):
        data_number, feature_number = X.shape

        outs = np.zeros((data_number, self.__units))
        for i in range(self.__units):
            outs[:, i] = kernel.gaussian_kernel(X, self.__centers[i].reshape(1, -1), self.__sigmas[i])
        return outs.dot(self.__weights)

    def predict(self, X):
        '''
        Parameters
        ----------
        X : shape (data_number, feature_number)
            Predicting data

        Returns
        -------
        y : shape (data_number, 1)
            Predicted class label per sample, 1 or 0
        '''
        return np.around(self.score(X))