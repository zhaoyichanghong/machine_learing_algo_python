import numpy as np
import distance

class LVQ:
    def fit(self, X, y, learning_rate, epochs):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data
        y : shape (n_samples,)
            Target values
        learning_rate : learning rate
        epochs : The number of epochs
        '''
        n_samples, n_features = X.shape

        classes = np.unique(y)
        n_classes = len(classes)

        self.__prototypes = np.zeros((n_classes, n_features))
        self.__prototypes_labels = np.zeros(n_classes)
        for i in range(n_classes):
            index_prototype = np.random.choice(np.flatnonzero(y == classes[i]), 1) 
            self.__prototypes[i] = X[index_prototype]
            self.__prototypes_labels[i] = y[index_prototype]
        
        for _ in range(epochs):            
            index = np.random.choice(n_samples, 1)

            distances = distance.euclidean_distance(X[index], self.__prototypes)
            nearest_index = np.argmin(distances)

            if self.__prototypes_labels[nearest_index] == y[index]:
                self.__prototypes[nearest_index] += learning_rate * (X[index] - self.__prototypes[nearest_index]).ravel()
            else:
                self.__prototypes[nearest_index] -= learning_rate * (X[index] - self.__prototypes[nearest_index]).ravel()
        
    def predict(self, X):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Predicting data

        Returns
        -------
        y : shape (n_samples,)
            Predicted class label per sample.
        '''
        distances = np.apply_along_axis(distance.euclidean_distance, 1, self.__prototypes, X).T
        return self.__prototypes_labels[np.argmin(distances, axis=1)]