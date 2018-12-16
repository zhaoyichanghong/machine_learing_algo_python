import numpy as np
import decision_tree

class GBDT:
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def __init__(self, loss):
        self.__models = []
        self.__alpha = []
        self.__loss = loss

    def fit(self, X, y, epochs, learning_rate):
        self.__learning_rate = learning_rate

        residual = y
        for _ in range(epochs):
            model = decision_tree.Cart('regression')
            model.fit(X, residual)
            self.__models.append(model)

            alpha = np.mean(residual / (model.predict(X) + 1e-8), axis=0)
            self.__alpha.append(alpha)

            residual = y - self.score(X)

    def predict(self, X, classes=None):
        if self.__loss == 'mse':
            return self.score(X)
        elif self.__loss == 'binary_crossentropy':
            return np.around(self.score(X))
        elif self.__loss == 'categorical_crossentropy':
            return classes[np.argmax(self.score(X), axis=1)].reshape((-1, 1))

    def score(self, X):
        h = 0
        for alpha, model in zip(self.__alpha, self.__models):
            h += self.__learning_rate * model.predict(X) * alpha

        if self.__loss == 'mse':
            return h
        elif self.__loss == 'binary_crossentropy':
            return self.__sigmoid(h)
        elif self.__loss == 'categorical_crossentropy':
            return self.__softmax(h)
        