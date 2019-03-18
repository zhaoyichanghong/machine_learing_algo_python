import numpy as np

class MinMaxScaler:
    def fit_transform(self, X):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data

        Returns
        -------
        X : shape (n_samples, n_features)
            The Training data min-max encoded.
        '''
        self.__min = np.min(X, axis=0)
        self.__max = np.max(X, axis=0)

        return self.transform(X)

    def transform(self, X):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Predicting data

        Returns
        -------
        X : shape (n_samples, n_features)
            The Predicting data min-max encoded.
        '''
        return (X - self.__min) / (self.__max - self.__min)

class StandardScaler:
    def fit_transform(self, X):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data

        Returns
        -------
        X : shape (n_samples, n_features)
            The Training data standard scaler encoded.
        '''
        self.__mean = np.mean(X, axis=0)
        self.__std = np.std(X, axis=0)

        return self.transform(X)

    def transform(self, X):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Predicting data

        Returns
        -------
        X : shape (n_samples, n_features)
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
        X : shape (n_samples, n_features)
            Training data

        Returns
        -------
        X : shape (n_samples, n_features)
            The Training data one hot encoded.
        '''
        self.__classes = np.unique(X)
        return self.transform(X)

    def transform(self, X):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Predicting data

        Returns
        -------
        X : shape (n_samples, n_features)
            The Predicting data one hot encoded.
        '''
        n_samples = X.shape[0]
        n_classes = len(self.classes)

        X_transformed = np.zeros((n_samples, n_classes))
        for i in range(n_classes):
            X_transformed[:, i] = (X == self.classes[i]).ravel()

        return X_transformed + 0

def bagging(n_samples, n_bags):
    '''
    Parameters
    ----------
    n_samples : The number of data
    n_bags : The number of bags

    Returns
    -------
    indexs : The indexes per bag included
    indexs_oob : The oob indexes per bag
    '''
    indexs = []
    indexs_oob = []
    for _ in range(n_bags):
        indexs.append(np.random.choice(range(n_samples), n_samples))
        indexs_oob.append(np.setdiff1d(range(n_samples), indexs[-1]))

    return indexs, indexs_oob