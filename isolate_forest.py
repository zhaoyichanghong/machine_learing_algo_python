import numpy as np
import treelib

class IsolateTree:
    class __data:
        def __init__(self):
            self.feature_split = None
            self.threshold_split = None
            self.n_samples = None

    def __get_normalization_factor(self, n):
        return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n

    def __init__(self):
        self.__tree = treelib.Tree()

    def __create_tree(self, parent, X):
        n_samples, n_features = X.shape

        data = self.__data()

        if n_samples == 0:
            data.n_samples = n_samples
            self.__tree.update_node(parent.identifier, data=data)
            return

        if self.__tree.depth() > self.__max_height or (X == X[0]).all():
            data.n_samples = n_samples
            self.__tree.update_node(parent.identifier, data=data)
            return

        data.feature_split = np.random.choice(n_features, 1)
        data.threshold_split = (max(X[:, data.feature_split]) - min(X[:, data.feature_split])) * np.random.random() + min(X[:, data.feature_split])

        self.__tree.update_node(parent.identifier, data=data)

        less_items = np.flatnonzero(X[:, data.feature_split] < data.threshold_split)
        greater_items = np.flatnonzero(X[:, data.feature_split] >= data.threshold_split)

        node = self.__tree.create_node('less ' + str(data.threshold_split), parent=parent)
        self.__create_tree(node, X[less_items])
        
        node = self.__tree.create_node('greater ' + str(data.threshold_split), parent=parent)
        self.__create_tree(node, X[greater_items])

    def fit(self, X):
        n_samples = X.shape[0]

        self.__c = self.__get_normalization_factor(n_samples)

        self.__max_height = np.round(np.log2(n_samples))

        root = self.__tree.create_node('root')
        self.__create_tree(root, X)

    def __get_path_length(self, x, node):
        if node.is_leaf():
            return self.__tree.depth(node.identifier) + (self.__get_normalization_factor(node.data.n_samples) if node.data.n_samples > 1 else 0)

        for child in self.__tree.children(node.identifier):
            if x[node.data.feature_split] < node.data.threshold_split and child.tag == 'less ' + str(node.data.threshold_split):
                return self.__get_path_length(x, child)
            elif x[node.data.feature_split] >= node.data.threshold_split and child.tag == 'greater ' + str(node.data.threshold_split):
                return self.__get_path_length(x, child)

    def get_abnormal_score(self, X):
        return 2 ** (-np.apply_along_axis(self.__get_path_length, 1, X, self.__tree.get_node(self.__tree.root)) / self.__c)

class IsolateForest:
    def __init__(self):
        self.__trees = []

    def fit(self, X, n_trees, n_subsamples):
        '''
        Parameters
        ----------
        X: shape (n_samples, n_features)
            Training data
        n_trees : The number of tree
        n_subsamples : The number of samples to draw from X to train each tree
        '''
        n_samples = X.shape[0]

        for _ in range(n_trees):
            sub_items = np.random.choice(n_samples, n_subsamples, replace=False)
            
            tree = IsolateTree()
            tree.fit(X[sub_items])

            self.__trees.append(tree)

    def predict(self, X):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Predicting data

        Returns
        -------
        y : shape (n_samples,)
            (a) if instances return abnormal score very close to 1, then they are deﬁnitely anomalies,
            (b) if instances haves much smaller than 0.5, then they are quite safe to be regarded as normal instances
            (c) if all the instances return abnormal score ≈ 0.5, then the entire sample does not really have any distinct anomaly.
        '''
        n_samples = X.shape[0]

        abnormal_scores = np.zeros((n_samples, len(self.__trees)))
        for i in range(len(self.__trees)):
            abnormal_scores[:, i] = self.__trees[i].get_abnormal_score(X)

        return np.mean(abnormal_scores, axis=1)