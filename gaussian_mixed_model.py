import numpy as np
import k_means
from scipy.stats import multivariate_normal
import distance

class gaussian_mixed_model:
    def fit_transform(self, X, cluster_number, epochs):
        data_number, feature_number = X.shape
        self.__cluster_number = cluster_number

        model = k_means.k_means()
        y = model.fit_transform(X, cluster_number, 1, distance.euclidean_distance)

        classes = np.unique(y)
        
        self.__pis = np.zeros((1, cluster_number))
        self.__means = np.zeros((cluster_number, feature_number))
        self.__sigma = np.zeros((cluster_number, feature_number, feature_number))
        for i in range(cluster_number):
            self.__pis[:, i] = np.mean(y == classes[i])

            indexes = np.where(y == classes[i])[0]

            self.__means[i] = np.mean(X[indexes], axis=0)
            self.__sigma[i] = np.cov(X[indexes].T)
        
        for _ in range(epochs):
            y_probs = self.score(X)
            
            number_classes = np.sum(y_probs, axis=0, keepdims=True)

            for i in range(cluster_number):
                self.__means[i] = np.sum(y_probs[:, i].reshape((-1, 1)) * X, axis=0) / number_classes[:, i]

                diff1 = (X - self.__means[i])[:,:,np.newaxis]
                diff2 = np.transpose(diff1, axes=(0, 2, 1)) * y_probs[:, i].reshape(-1, 1, 1)
                self.__sigma[i] = np.tensordot(diff1, diff2, axes=(0, 0)).reshape((feature_number, feature_number)) / number_classes[:, i]
                
                '''
                for j in range(data_number):
                    diff = (X[j] - self.__means[i]).reshape(-1, 1)
                    self.__sigma[i] += y_probs[j, i] * diff.dot(diff.T)
                self.__sigma[i] /= number_classes[:, i]
                '''

            self.__pis = number_classes / data_number

        return np.argmax(y_probs, axis=1)

    def score(self, X):
        data_number = X.shape[0]

        x_probs = np.zeros((data_number, self.__cluster_number))
        for i in range(self.__cluster_number):
            x_probs[:, i] = multivariate_normal.pdf(X, mean=self.__means[i], cov=self.__sigma[i])

        return self.__pis * x_probs / np.sum(self.__pis * x_probs, axis=1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.score(X), axis=1)