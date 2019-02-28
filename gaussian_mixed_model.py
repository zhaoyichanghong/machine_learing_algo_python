import numpy as np
from scipy.stats import multivariate_normal

class GMM:
    def fit(self, X, cluster_number, epochs):
        '''
        Parameters
        ----------
        X : shape (data_number, feature_number)
            Training data
        cluster_number : The number of clusters
        epochs : The number of epochs

        Returns
        -------
        y : shape (data_number, 1)
            Predicted cluster label per sample.
        '''
        data_number, feature_number = X.shape
        self.__cluster_number = cluster_number
        self.__phi = np.full(self.__cluster_number, 1 / self.__cluster_number)
        self.__means = X[np.random.choice(data_number, cluster_number)]
        self.__sigma = np.repeat(np.expand_dims(np.cov(X.T), axis=0), 3, axis=0)

        for _ in range(epochs):
            y_probs = self.score(X)
            
            number_classes = np.sum(y_probs, axis=0, keepdims=True)
            self.__phi = number_classes / data_number

            for i in range(self.__cluster_number):
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

        return self.predict(X)

    def score(self, X):
        data_number = X.shape[0]

        X_probs = np.zeros((data_number, self.__cluster_number))
        for i in range(self.__cluster_number):
            X_probs[:, i] = multivariate_normal.pdf(X, mean=self.__means[i], cov=self.__sigma[i])

        return self.__phi * X_probs / np.sum(self.__phi * X_probs, axis=1, keepdims=True)

    def predict(self, X):
        '''
        Parameters
        ----------
        X : shape (data_number, feature_number)
            Predicting data

        Returns
        -------
        y : shape (data_number, 1)
            Predicted cluster label per sample.
        '''
        return np.argmax(self.score(X), axis=1)