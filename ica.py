import numpy as np
import scipy.special
import pca

class Ica:
    def fit_transform(self, X, epochs, optimizer):
        '''
        Parameters
        ----------
        X : shape (data_number, feature_number)
            Training data
        epochs : The number of epochs
        optimizer : Optimize algorithm, see also optimizer.py

        Returns
        -------
        s : shape (data_number, feature_number)
            Predicted source per sample.
        '''
        data_number, feature_number = X.shape

        pca_model = pca.PCA(feature_number, True)
        X_whiten = pca_model.fit_transform(X)

        self.__W = np.random.rand(feature_number, feature_number)

        for _ in range(epochs):
            g_W = np.zeros_like(self.__W)
            for x in X_whiten:
                g_W += (1 - 2 * scipy.special.expit(self.__W.dot(x.T))).dot(x) + np.linalg.inv(self.__W.T)
            g_W /= data_number

            g_W = optimizer.optimize([g_W])[0]
            self.__W += g_W

        return X_whiten.dot(self.__W)
