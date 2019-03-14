import numpy as np
import treelib
import scipy.stats

class CART():
    class __data:
        def __init__(self):
            self.feature_split = None
            self.threshold_split = None
            self.feature_label_split = None
            self.result = None

    def __init__(self, mode='classification'):
        '''
        Parameters
        ----------
        mode : 'classification' or 'regression'
        '''
        self.__mode = mode
        self.__tree = treelib.Tree()

    def __get_gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        return 1 - np.sum((counts / np.sum(counts)) ** 2)      

    def __get_score(self, y_left, y_right):
        if self.__mode == 'regression':
            return -(np.std(y_left) + np.std(y_right))
        else:
            y_left_number = y_left.shape[0]
            y_right_number = y_right.shape[0]
            return (y_left_number * self.__get_gini(y_left) + y_right_number * self.__get_gini(y_right)) / (y_left_number + y_right_number)

    def __process_discrete(self, x, y):
        score_max = -np.inf
        for feature_label in np.unique(x):     
            left_items = np.flatnonzero(x == feature_label)
            right_items = np.flatnonzero(x != feature_label)
            score = self.__get_score(y[left_items], y[right_items])
            if score > score_max:
                score_max = score
                feature_label_split = feature_label
            
        return score_max, feature_label_split

    def __process_continuous(self, x, y):
        score_max = -np.inf
        x_sort = np.unique(np.sort(x))
        for j in range(len(x_sort) - 1):
            if y[j] == y[j+1]:
                continue
                
            threshold = (x_sort[j] + x_sort[j + 1]) / 2

            less_items = np.flatnonzero(x < threshold)
            greater_items = np.flatnonzero(x > threshold)
            score = self.__get_score(y[less_items], y[greater_items])
            if score > score_max:
                score_max = score
                threshold_split = threshold
            
        return score_max, threshold_split

    def __create_tree(self, parent, X, y):
        data_number, feature_number = X.shape

        if data_number == 0:
            return
        
        data = self.__data()
        if self.__mode == 'classification':
            data.result = max(set(y), key=y.tolist().count)
        else:
            data.result = np.mean(y, axis=0)

        if len(np.unique(y)) == 1 or (X == X[0]).all():
            self.__tree.update_node(parent.identifier, data=data)
            return

        score_max = -np.inf
        for i in range(feature_number):
            if len(np.unique(X[:, i])) == 1:
                continue

            try:
                feature = X[:, i].astype(float)
            except:
                score, feature_label = self.__process_discrete(X[:, i], y)
                threshold = None
            else:
                score, threshold = self.__process_continuous(feature, y)
                feature_label = None

            if score > score_max:
                score_max = score
                data.feature_split = i
                data.threshold_split = threshold
                data.feature_label_split = feature_label

        self.__tree.update_node(parent.identifier, data=data)
        if data.threshold_split:
            feature = X[:, data.feature_split].astype(float)

            less_items = np.flatnonzero(feature < data.threshold_split)
            greater_items = np.flatnonzero(feature > data.threshold_split)

            node = self.__tree.create_node('less ' + str(data.threshold_split), parent=parent)
            self.__create_tree(node, X[less_items], y[less_items])
            
            node = self.__tree.create_node('greater ' + str(data.threshold_split), parent=parent)
            self.__create_tree(node, X[greater_items], y[greater_items])
        elif data.feature_label_split:
            left_items = np.flatnonzero(X[:, data.feature_split] == data.feature_label_split)
            right_items = np.flatnonzero(X[:, data.feature_split] != data.feature_label_split)

            node = self.__tree.create_node('is ' + str(data.feature_label_split), parent=parent)
            self.__create_tree(node, X[left_items], y[left_items])
            
            node = self.__tree.create_node('not ' + str(data.feature_label_split), parent=parent)
            self.__create_tree(node, X[right_items], y[right_items])

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

    def prune_ccp(self):
        self.__tree.show()                        

    def __query(self, x, node):
        if node.is_leaf():
            return node.data.result

        for child in self.__tree.children(node.identifier):
            try:
                feature = x[node.data.feature_split].astype(float)
            except:
                if x[node.data.feature_split] == node.data.feature_label_split and child.tag == 'is ' + str(node.data.feature_label_split):
                    return self.__query(x, child)
                elif x[node.data.feature_split] != node.data.feature_label_split and child.tag == 'not ' + str(node.data.feature_label_split):
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