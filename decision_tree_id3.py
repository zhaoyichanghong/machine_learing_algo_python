import numpy as np
import treelib
import scipy.stats

class ID3():
    def __init__(self):
        self.__tree = treelib.Tree()

    def __get_entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        prob_classes = counts / np.sum(counts)
        return scipy.stats.entropy(prob_classes)

    def __create_tree(self, parent, X, y):
        data_number, feature_number = X.shape

        if data_number == 0 or len(np.unique(y)) == 1 or (X == X[0]).all():
            self.__tree.update_node(parent.identifier, data=max(set(y), key=y.tolist().count))
            return

        info_gain_max = -np.inf
        for i in range(feature_number):
            y_subs = [y[np.flatnonzero(X[:, i] == feature_label)] for feature_label in np.unique(X[:, i])]

            info_gain = self.__get_info_gain(y_subs, y)

            if info_gain > info_gain_max:
                info_gain_max = info_gain
                feature_split = i

        self.__tree.update_node(parent.identifier, data=feature_split)
        for feature_label in np.unique(X[:, feature_split]):
            node = self.__tree.create_node(feature_label, parent=parent)
            self.__create_tree(node, X[np.flatnonzero(X[:, feature_split] == feature_label)], y[np.flatnonzero(X[:, feature_split] == feature_label)])

    def __get_info_gain(self, y_subs, y):
        return self.__get_entropy(y) - sum([self.__get_entropy(y_sub) * len(y_sub) for y_sub in y_subs]) / len(y)

    def fit(self, X, y):
        '''
        Parameters
        ----------
        X : shape (data_number, feature_number)
            Training data
        y : shape (data_number)
            Target values, discrete value
        '''
        root = self.__tree.create_node('root')
        self.__create_tree(root, X, y)
        self.__tree.show()

    def __query(self, x, node):
        if node.is_leaf():
            return node.data

        feature_split = node.data
        for child in self.__tree.children(node.identifier):
            if x[feature_split] == child.tag:
                return self.__query(x, child)

    def predict(self, X):
        '''
        Parameters
        ----------
        X : shape (data_number, feature_number)
            Predicting data

        Returns
        -------
        y : shape (data_number,)
            Predicted class label per sample
        '''
        return np.apply_along_axis(self.__query, 1, X, self.__tree.get_node(self.__tree.root))