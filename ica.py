import numpy as np
import scipy.special
import pca

class Ica:
    def fit_transform(self, X, epochs, optimizer):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data
        epochs : The number of epochs
        optimizer : Optimize algorithm, see also optimizer.py

        Returns
        -------
        s : shape (n_samples, n_features)
            Predicted source per sample.
        '''
        n_samples, n_features = X.shape

        pca_model = pca.PCA(n_features, True)
        X_whiten = pca_model.fit_transform(X)

        self.__W = np.random.rand(n_features, n_features)

        for _ in range(epochs):
            g_W = np.zeros_like(self.__W)
            for x in X_whiten:
                g_W += (1 - 2 * scipy.special.expit(self.__W.dot(x.T))).dot(x) + np.linalg.inv(self.__W.T)
            g_W /= n_samples

            g_W = optimizer.optimize([g_W])[0]
            self.__W += g_W

        return X_whiten.dot(self.__W)
