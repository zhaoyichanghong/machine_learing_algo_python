import numpy as np
import matplotlib.pyplot as plt
import metrics
import regularizer
import scipy

class SoftmaxRegression:
    def __init__(self, debug=True):
        self.__debug = debug

    def fit(self, X, y, epochs, optimizer, regularizer=regularizer.Regularizer(0)):
        '''
        Parameters
        ----------
        X : shape (data_number, feature_number)
            Training data
        y : One-hot encoder, shape (data_number, class_number)
            Target values 
        epochs : The number of epochs
        optimizer : Optimize algorithm, see also optimizer.py
        regularizer : Regularize algorithm, see also regularizer.py
        '''
        data_number, feature_number = X.shape
        class_number = y.shape[1]

        self.__W = np.zeros((feature_number, class_number))
        self.__b = np.zeros((1, class_number))

        if self.__debug:
            accuracy = []
            loss = []
            
        for _ in range(epochs):
            h = self.score(X)

            g_W = X.T.dot(h - y) / data_number + regularizer.regularize(self.__W)
            g_b = np.mean(h - y, axis=0)
            g_W, g_b = optimizer.optimize([g_W, g_b])
            self.__W -= g_W
            self.__b -= g_b

            if self.__debug:
                h = self.score(X)
                loss.append(np.mean(-np.sum(y * np.log(h), axis=1)))
                accuracy.append(metrics.accuracy(np.argmax(y, axis=1), np.argmax(h, axis=1)))

        if self.__debug:
            _, ax_loss = plt.subplots()
            ax_loss.plot(loss, 'b')
            ax_accuracy = ax_loss.twinx()
            ax_accuracy.plot(accuracy, 'r')
            plt.show()

    def predict(self, X, classes):
        '''
        Parameters
        ----------
        X : shape (data_number, feature_number)
            Predicting data
        classes : shape (class_number,)
            The all labels

        Returns
        -------
        y : shape (data_number, 1)
            Predicted class label per sample.
        '''
        return classes[np.argmax(self.score(X), axis=1)].reshape((-1, 1))

    def score(self, X):
        '''
        Parameters
        ----------
        X : shape (data_number, feature_number)
            Predicting data

        Returns
        -------
        y : shape (data_number, class_number)
            Predicted score of class per sample.
        '''
        out = X.dot(self.__W) + self.__b
        return scipy.special.softmax(out - np.max(out), axis=1)