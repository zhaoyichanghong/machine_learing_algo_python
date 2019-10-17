import numpy as np
import timeit
from functools import reduce
import matplotlib.pyplot as plt
import scipy.special
import metrics
import regularizer
import weights_initializer

eta = 1e-8

class Conv2d:
    def __init__(self, filter_number, kernel_size, stride_size=1, input_size=0, optimizer=None, weights_initializer=weights_initializer.he_normal):
        '''
        Parameters
        ----------
        filter_number : filter number
        kernel_size : kernel size
        stride_size : stride size
        input_size : input shape
        optimizer : Optimize algorithm, see also optimizer.py
        weights_initializer : weight initializer, see also weights_initializer.py
        '''
        self.__filter_number = filter_number
        self.__input_size = input_size
        self.__kernel_size = kernel_size
        self.__stride_size = stride_size
        self.__optimizer = optimizer
        self.__weights_initializer = weights_initializer

    def init(self, input_size=0):
        if self.__input_size == 0:
            self.__input_size = input_size

        self.__input_channels, self.__input_h, self.__input_w = self.__input_size
        self.__padding_size = self.__kernel_size // 2
        self.__input_h += self.__padding_size * 2
        self.__input_w += self.__padding_size * 2

        self.__output_h = (self.__input_h - self.__kernel_size) // self.__stride_size + 1
        self.__output_w = (self.__input_w - self.__kernel_size) // self.__stride_size + 1
        self.__output_size = self.__output_h * self.__output_w

        self.output_size = (self.__filter_number, self.__output_h, self.__output_w)

        self.__W = self.__weights_initializer(self.__kernel_size ** 2 * self.__input_channels, self.__filter_number, (self.__filter_number, self.__input_channels, self.__kernel_size, self.__kernel_size))
        self.__b = np.zeros((self.__filter_number))

    def __img2col(self, img, input_channels):
        col = np.zeros((self.__batch_size, self.__output_size, self.__kernel_size ** 2 * input_channels))
        for h in range(0, self.__output_h):
            for w in range(0, self.__output_w):
                col[:, self.__output_w*h+w, :] = img[:, :, h*self.__stride_size:h*self.__stride_size+self.__kernel_size, w*self.__stride_size:w*self.__stride_size+self.__kernel_size].reshape(self.__batch_size, -1)
        
        return col

    def forward(self, X, mode):
        self.__batch_size = X.shape[0]
        self.__input_shape = X.shape

        X = np.pad(X[:, :], ((0, 0), (0, 0), (self.__padding_size, self.__padding_size), (self.__padding_size, self.__padding_size)), 'constant')

        self.__col = self.__img2col(X, self.__input_channels)
        output = self.__col.dot(self.__W.reshape(self.__filter_number, -1).T)

        return np.transpose(output, axes=(0, 2, 1)).reshape((self.__batch_size, self.__filter_number, self.__output_h, self.__output_w)) + self.__b[None, :, None, None]

    def optimize(self, residual):
        g_W = (np.tensordot(residual.reshape(self.__batch_size, self.__filter_number, -1), self.__col, axes=[[0,2], [0, 1]]) / self.__batch_size).reshape(self.__W.shape)
        g_b = np.mean(np.sum(residual, axis=(2, 3)), axis=0)
        
        g_W, g_b = self.__optimizer.optimize([g_W, g_b])
        
        self.__W -= g_W
        self.__b -= g_b

    def backward(self, residual):
        residual = np.pad(residual[:, :], ((0, 0), (0, 0), (self.__padding_size, self.__padding_size), (self.__padding_size, self.__padding_size)), 'constant')

        W = np.transpose(self.__W, axes=(1, 0, 2, 3))
        W = np.rot90(W, k=2, axes=(2, 3))

        col = self.__img2col(residual, self.__filter_number)
        residual = col.dot(W.reshape(self.__input_channels, -1).T)

        return np.transpose(residual, axes=(0, 2, 1)).reshape(self.__input_shape)

class MaxPool:
    def __init__(self, pool_size):
        '''
        Parameters
        ----------
        pool_size : pool size
        '''
        self.__pool_size = pool_size

    def init(self, input_size=0):
        self.__input_channels, self.__input_h, self.__input_w = input_size
        self.__output_h = (self.__input_h - self.__pool_size) // self.__pool_size + 1
        self.__output_w = (self.__input_w - self.__pool_size) // self.__pool_size + 1
        self.output_size = (self.__input_channels, self.__output_h, self.__output_w)

    def forward(self, X, mode):        
        self.__batch_size = X.shape[0]

        self.__output_index = np.zeros_like(X)
        for h in range(0, self.__output_h):
            for w in range(0, self.__output_w):
                output_index = np.argmax(X[:, :, h*self.__pool_size:h*self.__pool_size+self.__pool_size, w*self.__pool_size:w*self.__pool_size+self.__pool_size].reshape((self.__batch_size, self.__input_channels, -1)), axis=2)
                output_index_h, output_index_w = divmod(output_index, self.__pool_size)
                for i in range(self.__batch_size):
                    for j in range(self.__input_channels):
                        self.__output_index[i, j, h*self.__pool_size+output_index_h[i, j], w*self.__pool_size+output_index_w[i, j]] = 1

        return X.reshape(self.__batch_size, self.__input_channels, self.__output_h, self.__pool_size, self.__output_w, self.__pool_size).max(axis=(3,5))

    def backward(self, residual):
        residual = np.repeat(residual, repeats=self.__pool_size, axis=2)
        residual = np.repeat(residual, repeats=self.__pool_size, axis=3)
        return residual * self.__output_index

class MeanPool:
    def __init__(self, pool_size):
        '''
        Parameters
        ----------
        pool_size : pool size
        '''
        self.__pool_size = pool_size

    def init(self, input_size=0):
        self.__input_channels, self.__input_h, self.__input_w = input_size
        self.__output_h = (self.__input_h - self.__pool_size) // self.__pool_size + 1
        self.__output_w = (self.__input_w - self.__pool_size) // self.__pool_size + 1
        self.output_size = (self.__input_channels, self.__output_h, self.__output_w)

    def forward(self, X, mode):
        self.__batch_size = X.shape[0]
        
        H, W = self.__input_h // self.__pool_size, self.__input_w // self.__pool_size
        return X[:, :, :H*self.__pool_size, :W*self.__pool_size].reshape(self.__batch_size, self.__input_channels, H, self.__pool_size, W, self.__pool_size).mean(axis=(3, 5))

    def backward(self, residual):
        residual /= self.__pool_size ** 2
        residual = np.repeat(residual, repeats=self.__pool_size, axis=2)
        residual = np.repeat(residual, repeats=self.__pool_size, axis=3)
        return residual

class Flatten:
    def init(self, input_size=0):
        self.output_size = reduce(lambda i, j : i * j, input_size)

    def forward(self, X, mode):
        self.__input_shape = X.shape
        return X.reshape(self.__input_shape[0], -1)

    def backward(self, residual):
        return residual.reshape(self.__input_shape)

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
        loss : loss function including categorical_crossentropy, binary_crossentropy, mse, categorical_hinge
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