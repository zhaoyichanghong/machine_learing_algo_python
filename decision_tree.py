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

        if len(np.unique(y)) == 1:
            labels = y.flatten().tolist()
            return None, int(max(set(labels), key=labels.count))

        entropy = self.get_entropy(y)

        score_max = 0
        for i in range(feature_number):
            feature_sort = sorted(X[:, i])
            for n in range(data_number - 1):
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
        root['left']['node'], root['left']['label'] = self.__create_tree(X[left_items_boundary], y[left_items_boundary])
        root['right']['node'], root['right']['label'] = self.__create_tree(X[right_items_boundary], y[right_items_boundary])

        return root, None

    def fit(self, X, y):
        self.__root, _ = self.__create_tree(X, y)

    def __query(self, root, x):
        if x[root['feature']] < root['threshold']:
            if root['left']['node'] == None:
                return root['left']['label']
            else:
                return self.__query(root['left']['node'], x)
        else:
            if root['right']['node'] == None:
                return root['right']['label']
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

        if info_value == 0:
            return 0
        else:
            return info_gain / info_value
