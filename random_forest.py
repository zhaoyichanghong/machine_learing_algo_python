import numpy as np
import matplotlib.pyplot as plt
import preprocess
import metrics
import decision_tree_cart

class RandomForest:
    def __init__(self, mode='classification', debug=True):
        self.__trees = []
        self.__mode = mode
        self.__debug = debug

    def fit(self, X, y, n_trees, pick_n_features):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data
        y : shape (n_samples, 1)
            Target values, 1 or 0
        n_trees : The number of trees in the forest.
        pick_n_features : The number of features picked randomly
        '''
        n_samples, n_features = X.shape

        self.__indexs, self.__indexs_oob = preprocess.bagging(n_samples, n_trees)
        
        if self.__debug:
            accuracy = []

        for i in range(n_trees):
            features = np.random.choice(n_features, pick_n_features, replace=False)

            X_bag = X[self.__indexs[i]][:, features]
            y_bag = y[self.__indexs[i]]

            tree = decision_tree_cart.CART(self.__mode)
            tree.fit(X_bag, y_bag)

            self.__trees.append({'features':features, 'model':tree})
            
            if self.__debug:
                accuracy.append(self.__oob_verification(X, y))
        
        if self.__debug:
            plt.plot(accuracy)
            plt.show()

    def __oob_verification(self, X, y):
        n_samples = X.shape[0]
        n_trees = len(self.__trees)

        results = np.full((n_samples, n_trees), None)
        for i in range(n_trees):
            tree = self.__trees[i]['model']
            features = self.__trees[i]['features']
            X_bag_oob = X[self.__indexs_oob[i]][:, features]
            results[self.__indexs_oob[i], i] = tree.predict(X_bag_oob)

        y_pred = np.full_like(y, np.inf)
        for i in range(n_samples):
            if (results[i] == None).all():
                continue

            if self.__mode == 'regression':
                y_pred[i] = np.mean(results[i, np.flatnonzero(results[i] != None)])
            else:
                y_pred[i] = max(set(results[i, np.flatnonzero(results[i] != None)]), key=results[i, np.flatnonzero(results[i] != None)].tolist().count)

        if self.__mode == 'regression':
            return metrics.r2_score(y, y_pred)
        else:
            return metrics.accuracy(y, y_pred)

    def predict(self, X):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Predicting data

        Returns
        -------
        y : shape (n_samples, 1)
            Predicted value per sample
        '''
        n_samples = X.shape[0]

        results = np.empty((n_samples, 0))
        for tree in self.__trees:
            results = np.column_stack((results, tree['model'].predict(X[:, tree['features']])))
        
        if self.__mode == 'regression':
            y_pred = np.mean(results, axis=1)
        elif self.__mode == 'classification':
            pred = lambda result: max(set(result), key=result.tolist().count)
            y_pred = np.apply_along_axis(pred, 1, results)

        return y_pred

    def feature_selection(self, X, y):
        n_samples, n_features = X.shape

        model_score = self.__oob_verification(X, y)

        permutation = np.random.permutation(n_samples)
        scores = np.zeros(n_features)
        for i in range(n_features):
            X_tmp = X.copy()
            X_tmp[:, i] = X[permutation, i]
            scores[i] = self.__oob_verification(X_tmp, y)

        return model_score - scores