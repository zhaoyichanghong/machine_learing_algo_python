import numpy as np

class MinMaxScaler:
    def fit_transform(self, X):
        self.__min = np.min(X, axis=0)
        self.__max = np.max(X, axis=0)

        return self.transform(X)

    def transform(self, X):
        return (X - self.__min) / (self.__max - self.__min)

class StandardScaler:
    def fit_transform(self, X):
        self.__mean = np.mean(X, axis=0)
        self.__std = np.std(X, axis=0)

        return self.transform(X)

    def transform(self, X):
        return (X - self.__mean) / self.__std

class OneHot:
    @property
    def classes(self):
        return self.__classes

    def fit_transform(self, X):
        data_number = X.shape[0]
        self.__classes = np.unique(X)
        class_number = len(self.classes)

        X_transformed = np.zeros((data_number, class_number))
        for i in range(class_number):
            X_transformed[:, i] = (X == self.classes[i]).ravel()

        return X_transformed + 0

def bagging(data_number, bags_number):
    indexs = []
    indexs_oob = []
    for _ in range(bags_number):
        indexs.append(np.random.choice(range(data_number), data_number))
        indexs_oob.append(np.setdiff1d(range(data_number), indexs[-1]))

    return indexs, indexs_oob