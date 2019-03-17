import numpy as np

class MinMaxScaler:
    def fit_transform(self, X):
        '''
        Parameters
        ----------
        X : shape (data_number, feature_number)
            Training data

        Returns
        -------
        X : shape (data_number, feature_number)
            The Training data min-max encoded.
        '''
        self.__min = np.min(X, axis=0)
        self.__max = np.max(X, axis=0)

        return self.transform(X)

    def transform(self, X):
        '''
        Parameters
        ----------
        X : shape (data_number, feature_number)
            Predicting data

        Returns
        -------
        X : shape (data_number, feature_number)
            The Predicting data min-max encoded.
        '''
        return (X - self.__min) / (self.__max - self.__min)

class StandardScaler:
    def fit_transform(self, X):
        '''
        Parameters
        ----------
        X : shape (data_number, feature_number)
            Training data

        Returns
        -------
        X : shape (data_number, feature_number)
            The Training data standard scaler encoded.
        '''
        self.__mean = np.mean(X, axis=0)
        self.__std = np.std(X, axis=0)

        return self.transform(X)

    def transform(self, X):
        '''
        Parameters
        ----------
        X : shape (data_number, feature_number)
            Predicting data

        Returns
        -------
        X : shape (data_number, feature_number)
            The Predicting data standard scaler encoded.
        '''
        return (X - self.__mean) / (self.__std + 1e-8)

class OneHot:
    @property
    def classes(self):
        return self.__classes

    def fit_transform(self, X):
        '''
        Parameters
        ----------
        X : shape (data_number, feature_number)
            Training data

        Returns
        -------
        X : shape (data_number, feature_number)
            The Training data one hot encoded.
        '''
        self.__classes = np.unique(X)
        return self.transform(X)

    def transform(self, X):
        '''
        Parameters
        ----------
        X : shape (data_number, feature_number)
            Predicting data

        Returns
        -------
        X : shape (data_number, feature_number)
            The Predicting data one hot encoded.
        '''
        data_number = X.shape[0]
        class_number = len(self.classes)

        X_transformed = np.zeros((data_number, class_number))
        for i in range(class_number):
            X_transformed[:, i] = (X == self.classes[i]).ravel()

        return X_transformed + 0

def bagging(data_number, bags_number):
    '''
    Parameters
    ----------
    data_number : The number of data
    bags_number : The number of bags

    Returns
    -------
    indexs : The indexes per bag included
    indexs_oob : The oob indexes per bag
    '''
    indexs = []
    indexs_oob = []
    for _ in range(bags_number):
        indexs.append(np.random.choice(range(data_number), data_number))
        indexs_oob.append(np.setdiff1d(range(data_number), indexs[-1]))

    return indexs, indexs_oob