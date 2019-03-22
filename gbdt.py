import numpy as np
import decision_tree_cart
import scipy

class GBDT:
    def __init__(self, loss):
        self.__models = []
        self.__alpha = []
        self.__loss = loss

    def fit(self, X, y, epochs, learning_rate):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data
        y : shape (n_samples,)
            Target values 
        epochs : The number of epochs
        learning_rate : Learning rate
        '''
        self.__learning_rate = learning_rate

        residual = y
        for _ in range(epochs):
            model = decision_tree_cart.CART('regression')
            model.fit(X, residual)
            self.__models.append(model)

            alpha = np.mean(residual / (model.predict(X) + 1e-8), axis=0)
            self.__alpha.append(alpha)

            residual = y - self.score(X)

    def predict(self, X):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Predicting data

        Returns
        -------
        y : shape (n_samples,)
            Predicted value per sample.
        '''
        if self.__loss == 'regression':
            return self.score(X)
        elif self.__loss == 'classification':
            return np.around(self.score(X))

    def score(self, X):
        return self.__learning_rate * sum([model.predict(X) * alpha for alpha, model in zip(self.__alpha, self.__models)])
        