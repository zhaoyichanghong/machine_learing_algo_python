import numpy as np
import decision_stump

class discrete_adaboost:
    def fit(self, X, y, classifier_number):
        data_number = X.shape[0] 

        self.__alpha = np.zeros((classifier_number, 1))
        self.__classifiers = []
        
        w = np.full(data_number, 1 / data_number)

        for i in range(classifier_number):
            model = decision_stump.decision_stump()
            model.fit(X, y, w)
            h = model.predict(X)

            eta = np.sum(w[np.where(h != y)[0]]) / np.sum(w)
            beta = np.sqrt((1 - eta) / eta)
            w[np.where(h != y)[0]] *= beta
            w[np.where(h == y)[0]] /= beta

            self.__alpha[i] = np.log(beta)
            self.__classifiers.append(model)

    def predict(self, X):
        h = self.score(X)

        y_pred = np.ones_like(h)
        y_pred[np.where(h < 0)] = -1

        return y_pred

    def score(self, X):
        data_number = X.shape[0]
        classifier_number = len(self.__classifiers)

        h = 0
        for i in range(classifier_number):
            h += self.__alpha[i] * self.__classifiers[i].predict(X)

        return h