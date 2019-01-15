import numpy as np
import distance
import metrics
import matplotlib.pyplot as plt
import k_means

class RbfNet:
    def __init__(self, debug=True):
        self.__debug = debug

    def fit(self, X, y, cluster_number, learning_rate, epochs):
        data_number, feature_number = X.shape

        self.__cluster_number = cluster_number

        model = k_means.KMeans()
        model.fit_transform(X, self.__cluster_number, 10, distance.euclidean_distance)        
        self.__centers = model.centers
        
        self.__sigmas = np.ones((self.__cluster_number, 1))
        self.__weights = np.random.randn(self.__cluster_number, 1)
        
        if self.__debug:
            accuracy = []
            loss = []

        for _ in range(epochs):
            outs = np.zeros((data_number, self.__cluster_number))
            for i in range(self.__cluster_number):
                outs[:, i] = np.exp(-np.linalg.norm(X - self.__centers[i], axis=1) ** 2 / (2 * self.__sigmas[i] ** 2))
            h = outs.dot(self.__weights)
            residual = h - y
            
            g_centers = np.zeros_like(self.__centers)
            g_sigmas = np.zeros_like(self.__sigmas)
            g_weights = np.zeros_like(self.__weights)
            for i in range(self.__cluster_number):
                g_centers[i] = self.__weights[i] * np.mean(residual * outs[:, i].reshape((-1, 1)) * (X - self.__centers[i]), axis=0) / (self.__sigmas[i] ** 2)
                g_sigmas[i] = self.__weights[i] * np.mean(residual * outs[:, i].reshape((-1, 1)) * (np.linalg.norm(X - self.__centers[i], axis=1).reshape((-1, 1)) ** 2), axis=0) / (self.__sigmas[i] ** 3)
                g_weights[i] = np.mean(residual * outs[:, i].reshape((-1, 1)), axis=0)
            
            self.__centers -= learning_rate * g_centers
            self.__sigmas -= learning_rate * g_sigmas
            self.__weights -= learning_rate * g_weights

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

        outs = np.zeros((data_number, self.__cluster_number))
        for i in range(self.__cluster_number):
            outs[:, i] = np.exp(-np.linalg.norm(X - self.__centers[i], axis=1) ** 2 / (2 * self.__sigmas[i] ** 2))
        return outs.dot(self.__weights)

    def predict(self, X): 
        return np.around(self.score(X))