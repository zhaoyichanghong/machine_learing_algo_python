import numpy as np
import timeit
import matplotlib.pyplot as plt
import scipy
import metrics
import regularizer
import weights_initializer

eta = 1e-8

class Dense:
    def __init__(self, output_size, input_size=0, optimizer=None, regularizer=regularizer.Regularizer(0), weights_initializer=weights_initializer.he_normal):
        '''
        Parameters
        ----------
        output_size : output dimension
        input_size : input dimension
        optimizer : Optimize algorithm, see also optimizer.py
        regularizer : Regularize algorithm, see also regularizer.py
        weights_initializer : weight initializer, see also weights_initializer.py
        '''
        self.output_size = output_size
        self.__input_size = input_size
        self.__optimizer = optimizer
        self.__regularizer = regularizer
        self.__weights_initializer = weights_initializer

    def init(self, input_size=0):
        if self.__input_size == 0:
            self.__input_size = input_size

        self.__W = self.__weights_initializer(self.__input_size, self.output_size, (self.__input_size, self.output_size))
        self.__b = np.zeros(self.output_size)

    def forward(self, X, mode):
        self.__X = X
        return self.__X.dot(self.__W) + self.__b

    def backward(self, residual):
        return residual.dot(self.__W.T)

    def optimize(self, residual):
        batch_size = self.__X.shape[0]
        g_W = self.__X.T.dot(residual) / batch_size + self.__regularizer.regularize(self.__W)
        g_b = np.mean(residual, axis=0)

        g_W, g_b = self.__optimizer.optimize([g_W, g_b])

        self.__W -= g_W
        self.__b -= g_b

class Tanh:
    def init(self, input_size=0):
        self.output_size = input_size

    def forward(self, X, mode):
        self.__output = np.tanh(X)
        return self.__output

    def backward(self, residual):
        return (1 - self.__output ** 2) * residual

class Relu:
    def init(self, input_size=0):
        self.output_size = input_size
        
    def forward(self, X, mode):
        self.__X = X
        return np.maximum(self.__X, 0)

    def backward(self, residual):
        return (self.__X > 0) * residual

class Sigmoid:
    def forward(self, X, mode):
        return scipy.special.expit(X)

    def backward(self, residual):
        return residual

class Softmax:
    def forward(self, X, mode):
        return scipy.special.softmax(X - np.max(X, axis=1, keepdims=True), axis=1)

    def backward(self, residual):
        return residual

class NeuralNetwork:
    def __init__(self, loss, debug=True):
        '''
        Parameters
        ----------
        loss : loss function including categorical_crossentropy, binary_crossentropy, mse
        '''
        self.__debug = debug
        self.__layers = []
        self.__loss = loss

    def __get_residual(self, h, y):
        if self.__loss == 'binary_crossentropy' or self.__loss == 'mse' or self.__loss == 'categorical_crossentropy':
            return h - y
        elif self.__loss == 'categorical_hinge':
            batch_size = y.shape[0]
            correct_index = np.argmax(y, axis=1)
            residual = np.zeros_like(y)
            residual[h - h[range(batch_size), correct_index].reshape((-1, 1)) > 0] = 1
            residual[range(batch_size), correct_index] -= np.sum(residual, axis=1)

            return residual
           
    def __get_accuracy(self, h, y):
        if self.__loss == 'categorical_crossentropy' or self.__loss == 'categorical_hinge':
            return metrics.accuracy(np.argmax(h, axis=1), np.argmax(y, axis=1))
        elif self.__loss == 'binary_crossentropy':
            return metrics.accuracy(np.around(h), y)
        else:
            return 0

    def __get_loss(self, h, y):
        if self.__loss == 'binary_crossentropy':
            loss = -np.mean(np.sum(y * np.log(h + eta) + (1 - y) * np.log(1 - h + eta), axis=1))
        elif self.__loss == 'mse':
            loss = np.mean((h - y) ** 2)
        elif self.__loss == 'categorical_crossentropy':
            loss = -np.mean(np.sum(y * np.log(h + eta), axis=1))
        elif self.__loss == 'categorical_hinge':
            batch_size = y.shape[0]
            correct_index = np.argmax(y, axis=1)
            loss = np.mean(np.sum(np.maximum(0, h - h[range(batch_size), correct_index].reshape((-1, 1)) + 1), axis=1) - 1)

        return loss

    def __log(self, epoch, loss, elapse, accuracy):
        print('epochs: %d; loss: %f; elapse: %f; accuracy: %f' %(epoch, loss, elapse, accuracy))

    def __draw_figure(self, loss, accuracy):
        _, ax_loss = plt.subplots()
        ax_loss.plot(loss, 'r')

        ax_accuracy = ax_loss.twinx()
        ax_accuracy.plot(accuracy, 'b')

        plt.show()

    def __foreward(self, X, mode='fit'):
        for layer in self.__layers:
            X = layer.forward(X, mode)

        return X

    def __backward(self, residual):
        for layer in reversed(self.__layers):
            residual_backup = residual
            if layer != self.__layers[0]:
                residual = layer.backward(residual)
            if hasattr(layer, 'optimize'):
                layer.optimize(residual_backup)

    def add(self, layer):
        if hasattr(layer, 'init'):
            if len(self.__layers) > 0:
                layer.init(self.__layers[-1].output_size)
            else:
                layer.init()

        self.__layers.append(layer)

    def fit(self, X, y, batch_size, epochs):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Training data
        y : shape (n_samples, n_classes)
            Target values 
        batch_size : mini batch size
        epochs : The number of epochs
        '''
        data_number = X.shape[0]
        epoch = (data_number + batch_size - 1) // batch_size

        loss = []
        accuracy = []
        for _ in range(epochs):
            start_time = timeit.default_timer()

            permutation = np.random.permutation(data_number)
            X_epoch = X[permutation]
            y_epoch = y[permutation]

            for i in range(epoch):
                X_batch = X_epoch[batch_size*i : min(batch_size*(i+1), data_number)]
                y_batch = y_epoch[batch_size*i : min(batch_size*(i+1), data_number)]
                    
                h = self.__foreward(X_batch)
                residual = self.__get_residual(h, y_batch)
                self.__backward(residual)

                loss.append(self.__get_loss(h, y_batch))
                accuracy.append(self.__get_accuracy(h, y_batch))
                    
            self.__log(_, loss[-1], timeit.default_timer() - start_time, np.mean(accuracy))

        if self.__debug:
            self.__draw_figure(loss, accuracy)

    def predict(self, X, classes=None):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Predicting data
        classes : shape (n_classes,)
            The all labels

        Returns
        -------
        y : shape (n_samples, n_classes)
            Predicted class label per sample.
        '''
        if self.__loss == 'categorical_crossentropy' or self.__loss == 'categorical_hinge':
            return classes[np.argmax(self.score(X), axis=1)]
        elif self.__loss == 'binary_crossentropy':
            return np.around(self.score(X))
        elif self.__loss == 'mse':
            return self.score(X)

    def score(self, X):
        '''
        Parameters
        ----------
        X : shape (n_samples, n_features)
            Predicting data

        Returns
        -------
        y : shape (n_samples, n_classes)
            Predicted score of class per sample.
        '''
        return self.__foreward(X, 'predict')