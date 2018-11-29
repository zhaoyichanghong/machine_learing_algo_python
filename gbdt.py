import numpy as np
import decision_tree

class gbdt:
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def __init__(self, mode):
        self.__models = []
        self.__alpha = []
        self.__mode = mode

    def fit(self, X, y, epochs):
        h = np.zeros_like(y)

        for _ in range(epochs):
            residual = y - h

            model = decision_tree.cart('regression')
            model.fit(X, residual)
            self.__models.append(model)

            X_transform = model.predict(X)
            alpha = np.linalg.pinv(X_transform.T.dot(X_transform)).dot(X_transform.T).dot(residual)
            self.__alpha.append(alpha)

            residual = y - self.score(X)

    def predict(self, X, classes=None):
        if self.__mode == 'mse':
            return self.score()
        elif self.__mode == 'binary_crossentropy':
            return np.around(self.score(X))
        elif self.__mode == 'categorical_crossentropy':
            return classes[np.argmax(self.score(X), axis=1)].reshape((-1, 1))

    def score(self, X):
        h = 0

        for alpha, model in zip(self.__alpha, self.__models):
            h += model.predict(X).dot(alpha)

        if self.__mode == 'mse':
            return h
        elif self.__mode == 'binary_crossentropy':
            return self.__sigmoid(h)
        elif self.__mode == 'categorical_crossentropy':
            return self.__softmax(h)
        