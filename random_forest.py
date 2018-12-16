import numpy as np
import matplotlib.pyplot as plt
import preprocess
import metrics

class RandomForest:
    def __init__(self, tree_model, mode='classification', debug=True):
        self.__model = tree_model
        self.__trees = []
        self.__mode = mode
        self.__debug = debug

    def fit(self, X, y, trees_number, pick_feature_number):
        data_number, feature_number = X.shape

        self.__indexs, self.__indexs_oob = preprocess.bagging(data_number, trees_number)
        
        if self.__debug:
            accuracy = []

        for i in range(trees_number):
            features = np.random.choice(feature_number, pick_feature_number, replace=False)

            X_bag = X[self.__indexs[i]][:, features]
            y_bag = y[self.__indexs[i]]

            tree = self.__model(self.__mode)
            tree.fit(X_bag, y_bag)

            self.__trees.append({'features':features, 'model':tree})
            
            if self.__debug:
                accuracy.append(self.__oob_verification(X, y))
        
        if self.__debug:
            plt.plot(accuracy)
            plt.show()

    def __oob_verification(self, X, y):
        data_number = X.shape[0]
        trees_number = len(self.__trees)

        results = np.full((data_number, trees_number), np.inf)
        for i in range(trees_number):
            tree = self.__trees[i]['model']
            features = self.__trees[i]['features']
            X_bag_oob = X[self.__indexs_oob[i]][:, features]
            results[self.__indexs_oob[i], i] = tree.predict(X_bag_oob).ravel()

        y_pred = np.full_like(y, np.inf)
        for i in range(data_number):
            if (results[i] == np.inf).all():
                continue

            if self.__mode == 'regression':
                y_pred[i] = np.mean(results[i, np.where(results[i] != np.inf)])
            else:
                y_pred[i] = np.argmax(np.bincount(results[i][np.where(results[i] != np.inf)].astype(int)))

        if self.__mode == 'regression':
            return metrics.r2_score(y, y_pred)
        else:
            return metrics.accuracy(y, y_pred)

    def predict(self, X):
        data_number = X.shape[0]

        results = np.empty((data_number, 0))
        for tree in self.__trees:
            results = np.column_stack((results, tree['model'].predict(X[:, tree['features']])))
        
        if self.__mode == 'regression':
            y_pred = np.mean(results, axis=1, keepdims=True)
        elif self.__mode == 'classification':
            pred = lambda result: np.argmax(np.bincount(result.astype(int)))
            y_pred = np.apply_along_axis(pred, 1, results).reshape((-1, 1))

        return y_pred

    def feature_selection(self, X, y):
        data_number, feature_number = X.shape

        model_score = self.__oob_verification(X, y)

        permutation = np.random.permutation(data_number)
        scores = np.zeros(feature_number)
        for i in range(feature_number):
            X_tmp = X.copy()
            X_tmp[:, i] = X[permutation, i]
            scores[i] = self.__oob_verification(X_tmp, y)

        return model_score - scores