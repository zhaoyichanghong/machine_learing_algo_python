import numpy as np
import collections

class decision_tree():
    def __tree(self): 
        return collections.defaultdict(self.__tree)

    def get_entropy(self, y):
        data_number = y.shape[0]

        items_number = []
        for key, value in collections.Counter(y.flatten()).items():
            items_number.append(value)
        items_number = np.array(items_number)

        return -np.sum(items_number / data_number * np.log2(items_number / data_number))

    def __create_tree(self, X, y):
        data_number, feature_number = X.shape

        if data_number == 0:
            return None, None

        if len(np.unique(y)) == 1 or np.allclose(np.mean(X, axis=0), X[0]):
            if hasattr(self, 'mode') and self.mode == 'regression':
                return None, np.mean(y)
            else:
                labels = y.flatten().tolist()
                return None, int(max(set(labels), key=labels.count))

        entropy = self.get_entropy(y)

        score_max = -np.inf
        for i in range(feature_number):
            feature_sort = np.unique(sorted(X[:, i]))
            for n in range(len(feature_sort) - 1):
                threshold = (feature_sort[n] + feature_sort[n + 1]) / 2

                left_items = np.where(X[:, i] < threshold)[0]
                right_items = np.where(X[:, i] >= threshold)[0]

                score = self.get_score(y[left_items], y[right_items], entropy)

                if score > score_max:
                    score_max = score
                    feature_boundary = i
                    threshold_boundary = threshold
                    left_items_boundary = left_items
                    right_items_boundary = right_items

        root = self.__tree()
        root['feature'] = feature_boundary
        root['threshold'] = threshold_boundary
        root['left']['node'], root['left']['result'] = self.__create_tree(X[left_items_boundary], y[left_items_boundary])
        root['right']['node'], root['right']['result'] = self.__create_tree(X[right_items_boundary], y[right_items_boundary])

        return root, None

    def fit(self, X, y):
        self.__root, _ = self.__create_tree(X, y)

    def __query(self, root, x):
        if x[root['feature']] < root['threshold']:
            if root['left']['node'] == None:
                return root['left']['result']
            else:
                return self.__query(root['left']['node'], x)
        else:
            if root['right']['node'] == None:
                return root['right']['result']
            else:
                return self.__query(root['right']['node'], x)

    def predict(self, X):
        data_number = X.shape[0]
        
        y_predict = []
        for n in range(data_number):
            y_predict.append(self.__query(self.__root, X[n]))

        return np.array(y_predict).reshape((-1, 1))

class id3(decision_tree):
    def get_score(self, y_left, y_right, entropy):
        y_left_number = y_left.shape[0]
        y_right_number = y_right.shape[0]
        data_number = y_left_number + y_right_number

        return entropy - (y_left_number / data_number * self.get_entropy(y_left) + y_right_number / data_number * self.get_entropy(y_right))

class c4_5(decision_tree):
    def get_score(self, y_left, y_right, entropy):
        y_left_number = y_left.shape[0]
        y_right_number = y_right.shape[0]
        data_number = y_left_number + y_right_number

        info_gain = entropy - (y_left_number / data_number * self.get_entropy(y_left) + y_right_number / data_number * self.get_entropy(y_right))

        if y_left_number == 0:
            info_value = - y_right_number / data_number * np.log2(y_right_number / data_number)
        elif y_right_number == 0:
            info_value = -y_left_number / data_number * np.log2(y_left_number / data_number)
        else:
            info_value = -y_left_number / data_number * np.log2(y_left_number / data_number) - y_right_number / data_number * np.log2(y_right_number / data_number)

        return info_gain / info_value

class cart(decision_tree):
    def __init__(self, mode='classification'):
        self.mode = mode

    def __get_gini(self, y):
        data_number = y.shape[0]

        items_number = []
        for key, value in collections.Counter(y.flatten()).items():
            items_number.append(value)
        items_number = np.array(items_number)

        return 1 - np.sum((items_number / data_number) ** 2)

    def get_score(self, y_left, y_right, entropy):
        y_left_number = y_left.shape[0]
        y_right_number = y_right.shape[0]
        data_number = y_left_number + y_right_number

        if self.mode == 'regression':
            return -(np.std(y_left) + np.std(y_right))
        else:
            return -(y_left_number / data_number * self.__get_gini(y_left) + y_right_number / data_number * self.__get_gini(y_right))