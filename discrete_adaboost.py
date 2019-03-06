import numpy as np
import decision_stump

class DiscreteAdaboost:
    def fit(self, X, y, n_estimators):
        '''
        Parameters
        ----------
        X : shape (data_number, feature_number)
            Training data
        y : shape (data_number, 1)
            Target values, 1 or -1
        n_estimators : The number of estimators at which boosting is terminated
        '''
        data_number = X.shape[0] 

        self.__alpha = np.zeros((n_estimators, 1))
        self.__estimators = []
        
        w = np.full(data_number, 1 / data_number)
        for i in range(n_estimators):
            model = decision_stump.DecisionStump()
            model.fit(X, y, w)
            h = model.predict(X)

            eta = np.sum(w[np.flatnonzero(h != y)]) / np.sum(w)
            beta = np.sqrt((1 - eta) / (eta + 1e-8))
            w[np.flatnonzero(h != y)] *= beta
            w[np.flatnonzero(h == y)] /= beta

            self.__alpha[i] = np.log(beta)
            self.__estimators.append(model)

    def predict(self, X):
        '''
        Parameters
        ----------
        X : shape (data_number, feature_number)
            Predicting data

        Returns
        -------
        y : shape (data_number, 1)
            Predicted class label per sample, 1 or -1
        '''
        h = self.score(X)

        y_pred = np.ones_like(h)
        y_pred[np.flatnonzero(h < 0)] = -1

        return y_pred

    def score(self, X):
        return sum([alpha * classifier.predict(X) for alpha, classifier in zip(self.__alpha, self.__estimators)])