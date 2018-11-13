import numpy as np
import preprocess
import decision_tree

class random_forest:
    def __init__(self, tree_model, mode='classification'):
        self.__model = tree_model
        self.__trees = []
        self.__mode = mode

    def fit(self, X, y, trees_number, pick_feature_number):
        feature_number = X.shape[1]

        bags, bags_oob = preprocess.bagging(X, y, trees_number)
        
        for i in range(trees_number):
            features = np.random.choice(feature_number, pick_feature_number, replace=False)

            X_bag = bags[i]['X'][:, features]
            y_bag = bags[i]['y']

            self.__trees.append({'features':features, 'model':self.__model(self.__mode)})
            self.__trees[-1]['model'].fit(X_bag, y_bag)          

    def predict(self, X):
        data_number = X.shape[0]

        results = np.empty((data_number, 0))
        for tree in self.__trees:
            results = np.column_stack((results, tree['model'].predict(X[:, tree['features']])))
        
        if self.__mode == 'regression':
            y_pred = np.mean(results, axis=1).reshape((-1, 1))
        else:
            y_pred = np.zeros((data_number, 1))
            for i in range(data_number):
                y_pred[i] = int(max(set(results[i]), key=results[i].tolist().count))

        return y_pred