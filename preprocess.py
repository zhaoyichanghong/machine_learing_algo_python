import numpy as np

class min_max_scaler:
    def fit_transform(self, X):
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)

        return (X - self.min) / (self.max - self.min)

    def transform(self, X):
        return (X - self.min) / (self.max - self.min)

class standard_scaler:
    def fit_transform(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

        return (X - self.mean) / self.std

    def transform(self, X):
        return (X - self.mean) / self.std

class one_hot:
    def fit_transform(self, X):
        data_number = X.shape[0]
        self.classes = np.unique(X)
        class_number = len(self.classes)

        X_transformed = np.zeros((data_number, class_number))
        for i in range(class_number):
            X_transformed[:, i] = (X == self.classes[i]).flatten()

        return X_transformed + 0

def bagging(X, y, bags_number):
    data_number = X.shape[0]

    bags = []
    bags_oob = []
    for _ in range(bags_number):
        bags_index = np.random.choice(range(data_number), data_number)
        bags_oob_index = list(set(range(data_number)).difference(set(bags_index)))

        bags.append({'X':X[bags_index], 'y':y[bags_index]})
        bags_oob.append({'X':X[bags_oob_index], 'y':y[bags_oob_index]})

    return bags, bags_oob