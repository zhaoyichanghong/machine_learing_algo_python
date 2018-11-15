import numpy as np
import matplotlib.pyplot as plt
import preprocess
import decision_tree
import metrics

class random_forest:
    def __init__(self, tree_model, mode='classification', debug=True):
        self.__model = tree_model
        self.__trees = []
        self.__mode = mode
        self.__debug = debug

    def fit(self, X, y, trees_number, pick_feature_number):
        data_number, feature_number = X.shape

        indexs, indexs_oob = preprocess.bagging(data_number, trees_number)
        
        if self.__debug:
            accuracy = []
            results = np.full((data_number, trees_number), 0xFFFF)

        for i in range(trees_number):
            features = np.random.choice(feature_number, pick_feature_number, replace=False)

            X_bag = X[indexs[i]][:, features]
            y_bag = y[indexs[i]]

            tree = self.__model(self.__mode)
            tree.fit(X_bag, y_bag)

            self.__trees.append({'features':features, 'model':tree})
            
            if self.__debug:
                X_bag_oob = X[indexs_oob[i]][:, features]
                results[indexs_oob[i], i] = tree.predict(X_bag_oob).flatten()

                y_pred = np.full((data_number, 1), 0xFFFF)
                if self.__mode == 'regression':
                    for n in range(data_number):
                        valid_items = np.where(results[n, :len(self.__trees)] != 0xFFFF)[0]
                        if np.size(valid_items) != 0:
                            y_pred[n] = np.mean(results[n, valid_items])
                    
                    pred_items = np.where(y_pred != 0xFFFF)[0]
                    accuracy.append(metrics.r2_score(y[pred_items], y_pred[pred_items]))
                elif self.__mode == 'classification':
                    for n in range(data_number):
                        valid_items = np.where(results[n, :len(self.__trees)] != 0xFFFF)[0]
                        if np.size(valid_items) != 0:
                            y_pred[n] = np.argmax(np.bincount(results[n, valid_items].astype(int)))

                    pred_items = np.where(y_pred != 0xFFFF)[0]
                    accuracy.append(metrics.accuracy(y[pred_items], y_pred[pred_items]))
        
        if self.__debug:
            plt.plot(accuracy)
            plt.show()

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