import numpy as np
import collections
import metrics

class DecisionTree():
    def __is_leaf(self, node):
        if node['left'] is None and node['right'] is None:
            return True
        else:
            return False

    def __init__(self):
        self.__prune_nodes = []

    def __node(self):
        root = collections.defaultdict(self.__node)
        root['left'] = None
        root['right'] = None
        return root

    def _get_entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        prob_classes = counts / np.sum(counts)
        return -np.sum(prob_classes * np.log2(prob_classes))

    def __create_tree(self, root, X, y):
        data_number, feature_number = X.shape

        if data_number == 0:
            return

        if hasattr(self, '_mode') and self._mode == 'regression':
            root['result'] = np.mean(y, axis=0)
            root['error_number'] = np.sum((y - root['result']) ** 2)
        else:
            root['result'] = np.argmax(np.bincount(y.flatten().astype(int)))
            root['error_number'] = np.sum(y != root['result'])
            root['data_number'] = data_number

        if len(np.unique(y)) == 1 or np.isclose(X, X[0]).all():
            return

        entropy = self._get_entropy(y)

        score_max = -np.inf
        for i in range(feature_number):
            feature_sort = np.unique(np.sort(X[:, i]))
            for n in range(len(feature_sort) - 1):
                threshold = (feature_sort[n] + feature_sort[n + 1]) / 2

                left_items = np.flatnonzero(X[:, i] < threshold)
                right_items = np.flatnonzero(X[:, i] >= threshold)

                score = self.get_score(y[left_items], y[right_items], entropy)

                if score > score_max:
                    score_max = score
                    feature_split = i
                    threshold_split = threshold
                    left_items_split = left_items
                    right_items_split = right_items

        root['feature'] = feature_split
        root['threshold'] = threshold_split

        root['left'] = self.__node()
        root['right'] = self.__node()
        self.__create_tree(root['left'], X[left_items_split], y[left_items_split])
        self.__create_tree(root['right'], X[right_items_split], y[right_items_split])

    def fit(self, X, y):
        self.__root = self.__node()
        self.__create_tree(self.__root, X, y)

    def __query(self, x, root):
        if self.__is_leaf(root) or (root in self.__prune_nodes):
            return root['result']

        if x[root['feature']] < root['threshold']:
            return self.__query(x, root['left'])
        else:
            return self.__query(x, root['right'])

    def predict(self, X):
        return np.apply_along_axis(self.__query, 1, X, self.__root).reshape(-1, 1)
    
    def __traversal_bottom2top(self, root):
        queue = [root]
        nodes = []

        while queue:
            length = len(queue)
            for _ in range(length):
                node = queue.pop(0)
                if not self.__is_leaf(node):
                    nodes.append(node)
                    queue.append(node['left'])
                    queue.append(node['right'])
        
        return nodes[::-1]

    def __rep(self, X, y):
        nodes = self.__traversal_bottom2top(self.__root)

        acc = metrics.accuracy(y, self.predict(X))
        for node in nodes:
            self.__prune_nodes.append(node)
            acc_prune = metrics.accuracy(y, self.predict(X))

            if acc_prune < acc:
                self.__prune_nodes.pop()
            else:
                acc = acc_prune

    def __compute_leaves(self, root):
        if self.__is_leaf(root):
            return 1, root['error_number']
        elif hasattr(self, '__prune_sequence') and (root in self.__prune_sequence):
            return 1, root['error_number']
        else:
            return list(map(lambda x: x[0] + x[1], zip(self.__compute_leaves(root['left']), self.__compute_leaves(root['right']))))

    def __traversal(self, root, func):
        if self.__is_leaf(root) or (root in self.__prune_sequence):
            return

        func(root)

        self.__traversal(root['left'], func)
        self.__traversal(root['right'], func)

    def __ccp(self, X, y):
        def compute_costs(node):
            leaves_count, leaves_error = self.__compute_leaves(node)
        
            cost = (node['error_number'] - leaves_error) / (leaves_count - 1)
            self.__costs.append([node, cost, leaves_count])

        self.__prune_sequence = []
        while True:
            self.__costs = []
            self.__traversal(self.__root, compute_costs)
            if not self.__costs:
                break

            self.__costs = np.array(self.__costs)
            self.__costs = self.__costs[self.__costs[:, 1].argsort()]
            min_costs = self.__costs[np.flatnonzero(self.__costs[:, 1] == self.__costs[0, 1])]
            min_costs = min_costs[min_costs[:, 2].argsort()]
            self.__prune_sequence.append(min_costs[-1][0])
                
        accuracy_max = 0
        prune_nodes_final = []
        for i in range(len(self.__prune_sequence) + 1):
            self.__prune_nodes = self.__prune_sequence[:i]
            accuracy = metrics.accuracy(y, self.predict(X))

            if accuracy >= accuracy_max:
                accuracy_max = accuracy
                prune_nodes_final = self.__prune_nodes

        self.__prune_nodes = prune_nodes_final
    
    def __pep(self):
        nodes = self.__traversal_bottom2top(self.__root)

        for node in nodes[::-1]:
            leaves_count, leaves_error = self.__compute_leaves(node)

            error = (leaves_error + leaves_count * 0.5) / node['data_number']
            std = np.sqrt(error * (1 - error) * node['data_number'])

            if error * node['data_number'] + std > node['error_number'] + 0.5:
                #self.__traversal(node, lambda n: nodes.remove(n))
                self.__prune_nodes.append(node)
    
    def prune(self, X=None, y=None, solver='ccp'):
        if solver == 'rep':
            self.__rep(X, y)
        elif solver == 'ccp':
            self.__ccp(X, y)
        elif solver == 'pep':
            self.__pep()
        print(self.__prune_nodes)

class ID3(DecisionTree):
    def get_score(self, y_left, y_right, entropy):
        y_left_number = y_left.shape[0]
        y_right_number = y_right.shape[0]
        data_number = y_left_number + y_right_number

        return entropy - (y_left_number / data_number * self._get_entropy(y_left) + y_right_number / data_number * self._get_entropy(y_right))

class C4_5(DecisionTree):
    def get_score(self, y_left, y_right, entropy):
        y_left_number = y_left.shape[0]
        y_right_number = y_right.shape[0]
        data_number = y_left_number + y_right_number

        info_gain = entropy - (y_left_number / data_number * self._get_entropy(y_left) + y_right_number / data_number * self._get_entropy(y_right))

        if y_left_number == 0:
            info_value = -y_right_number / data_number * np.log2(y_right_number / data_number)
        elif y_right_number == 0:
            info_value = -y_left_number / data_number * np.log2(y_left_number / data_number)
        else:
            info_value = -y_left_number / data_number * np.log2(y_left_number / data_number) - y_right_number / data_number * np.log2(y_right_number / data_number)

        return info_gain / info_value

class Cart(DecisionTree):
    def __init__(self, mode='classification'):
        super().__init__()
        self._mode = mode

    def __get_gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        return 1 - np.sum((counts / np.sum(counts)) ** 2)

    def get_score(self, y_left, y_right, entropy):
        if self._mode == 'regression':
            return -(np.std(y_left) + np.std(y_right))
        else:
            y_left_number = y_left.shape[0]
            y_right_number = y_right.shape[0]
            data_number = y_left_number + y_right_number
            return -(y_left_number / data_number * self.__get_gini(y_left) + y_right_number / data_number * self.__get_gini(y_right))
