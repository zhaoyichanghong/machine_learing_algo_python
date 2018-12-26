import numpy as np
import decision_stump

class DiscreteAdaboost:
    def fit(self, X, y, classifier_number):
        data_number = X.shape[0] 

        self.__alpha = np.zeros((classifier_number, 1))
        self.__classifiers = []
        
        w = np.full(data_number, 1 / data_number)

        for i in range(classifier_number):
            model = decision_stump.DecisionStump()
            model.fit(X, y, w)
            h = model.predict(X)

            eta = np.sum(w[np.flatnonzero(h != y)]) / np.sum(w)
            beta = np.sqrt((1 - eta) / (eta + 1e-8))
            w[np.flatnonzero(h != y)] *= beta
            w[np.flatnonzero(h == y)] /= beta

            self.__alpha[i] = np.log(beta)
            self.__classifiers.append(model)

    def predict(self, X):
        h = self.score(X)

        y_pred = np.ones_like(h)
        y_pred[np.flatnonzero(h < 0)] = -1

        return y_pred

    def score(self, X):
        h = 0
        for alpha, classifier in zip(self.__alpha, self.__classifiers):
            h += alpha * classifier.predict(X)

        return h