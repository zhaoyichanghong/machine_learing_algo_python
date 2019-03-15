import numpy as np
import treelib
import scipy.stats

class C45():
    class __data:
        def __init__(self):
            self.feature_split = None
            self.threshold_split = None
            self.data_number = None
            self.error_number = None
            self.result = None

    def __init__(self):
        self.__tree = treelib.Tree()

    def __get_entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        prob_classes = counts / np.sum(counts)
        return scipy.stats.entropy(prob_classes)

    def __get_info_gain(self, X_subs, y_subs, y):
        return self.__get_entropy(y) - sum([self.__get_entropy(y_sub) * len(y_sub) for y_sub in y_subs]) / len(y)

    def __get_info_gain_ratio(self, X_subs, y_subs, y):
        info_gain = self.__get_info_gain(X_subs, y_subs, y)
        if info_gain == 0:
            return 0

        return info_gain / self.__get_entropy(X_subs)          

    def __process_discrete(self, x, y):
        y_subs = [y[np.flatnonzero(x == feature_label)] for feature_label in np.unique(x)]
        return self.__get_info_gain_ratio(x, y_subs, y), None

    def __process_continuous(self, x, y):
        info_gain_max = -np.inf
        x_sort = np.unique(np.sort(x))
        for j in range(len(x_sort) - 1):
            threshold = (x_sort[j] + x_sort[j + 1]) / 2

            less_items = np.flatnonzero(x < threshold)
            greater_items = np.flatnonzero(x > threshold)
            y_subs = [y[less_items], y[greater_items]]
            X_subs = np.append(np.zeros(len(less_items)), np.ones(len(greater_items)))

            info_gain = self.__get_info_gain(X_subs, y_subs, y)
            if info_gain > info_gain_max:
                info_gain_max = info_gain
                threshold_split = threshold
                info_gain_ratio = self.__get_info_gain_ratio(X_subs, y_subs, y)
            
        return info_gain_ratio, threshold_split

    def __create_tree(self, parent, X, y):
        data_number, feature_number = X.shape

        if data_number == 0:
            return
        
        data = self.__data()
        data.data_number = data_number
        data.result = max(set(y), key=y.tolist().count)
        data.error_number = sum(y != data.result)

        if len(np.unique(y)) == 1 or (X == X[0]).all():
            self.__tree.update_node(parent.identifier, data=data)
            return

        info_gain_ratio_max = -np.inf
        for i in range(feature_number):
            if len(np.unique(X[:, i])) == 1:
                continue

            try:
                feature = X[:, i].astype(float)
            except:
                info_gain_ratio, threshold = self.__process_discrete(X[:, i], y) 
            else:
                info_gain_ratio, threshold = self.__process_continuous(feature, y)

            if info_gain_ratio > info_gain_ratio_max:
                info_gain_ratio_max = info_gain_ratio
                data.feature_split = i
                data.threshold_split = threshold

        self.__tree.update_node(parent.identifier, data=data)
        if data.threshold_split:
            feature = X[:, data.feature_split].astype(float)

            less_items = np.flatnonzero(feature < data.threshold_split)
            greater_items = np.flatnonzero(feature > data.threshold_split)

            node = self.__tree.create_node('less ' + str(data.threshold_split), parent=parent)
            self.__create_tree(node, X[less_items], y[less_items])
            
            node = self.__tree.create_node('greater ' + str(data.threshold_split), parent=parent)
            self.__create_tree(node, X[greater_items], y[greater_items])
        else:
            for feature_label in np.unique(X[:, data.feature_split]):
                node = self.__tree.create_node(feature_label, parent=parent)
                self.__create_tree(node, X[np.flatnonzero(X[:, data.feature_split] == feature_label)], y[np.flatnonzero(X[:, data.feature_split] == feature_label)])

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

    def prune_pep(self):
        for level in reversed(range(self.__tree.depth())):
            for node in self.__tree.all_nodes():
                if not self.__tree.contains(node.identifier):
                    continue

                if self.__tree.level(node.identifier) == level and not node.is_leaf():
                    leaves_number = len(self.__tree.leaves(node.identifier))
                    leaves_error = sum([leaf.data.error_number for leaf in self.__tree.leaves(node.identifier)])
                    error = (leaves_error + leaves_number * 0.5) / node.data.data_number
                    std = np.sqrt(error * (1 - error) * node.data.data_number)
                    if leaves_error + leaves_number * 0.5 + std > node.data.error_number + 0.5:
                        for child in self.__tree.children(node.identifier):
                            self.__tree.remove_node(child.identifier)

        self.__tree.show()                        

    def __query(self, x, node):
        if node.is_leaf():
            return node.data.result

        for child in self.__tree.children(node.identifier):
            try:
                feature = x[node.data.feature_split].astype(float)
            except:
                if x[node.data.feature_split] == child.tag:
                    return self.__query(x, child)
            else:
                if feature < node.data.threshold_split and child.tag == 'less ' + str(node.data.threshold_split):
                    return self.__query(x, child)
                elif feature > node.data.threshold_split and child.tag == 'greater ' + str(node.data.threshold_split):
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