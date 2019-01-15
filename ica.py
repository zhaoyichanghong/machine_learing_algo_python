import numpy as np
import pca

class Ica:
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit_transform(self, X, epochs, learning_rate):
        data_number, feature_number = X.shape

        pca_model = pca.PCA(feature_number, True)
        X_whiten = pca_model.fit_transform(X)

        self.__W = np.random.rand(feature_number, feature_number)

        for _ in range(epochs):
            g_W = np.zeros_like(self.__W)
            for x in X_whiten:
                g_W += (1 - 2 * self.__sigmoid(self.__W.dot(x.T))).dot(x) + np.linalg.inv(self.__W.T)

            g_W /= data_number
            self.__W += learning_rate * g_W

        return X_whiten.dot(self.__W)
