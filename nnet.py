import numpy as np
import matplotlib.pyplot as plt
import timeit
import metrics
import optimizer
import threading
import random
import math
from functools import reduce
import regularizer
import weights_initializer

class Conv1d:
    def __init__(self, filter_number, kernel_size, stride_size=1, padding='same',  input_size=0, weights_initializer=weights_initializer.he_normal):
        self.__filter_number = filter_number
        self.__input_size = input_size
        self.__kernel_size = kernel_size
        self.__stride_size = stride_size
        self.__padding = padding
        self.__weights_initializer = weights_initializer

    def init(self, optimizer, learning_rate, input_size=0):
        if self.__input_size == 0:
            self.__input_size = input_size

        self.__input_timesteps, self.__input_channels = self.__input_size

        if self.__padding == 'same':
            self.__padding_size = (math.ceil(self.__input_timesteps / self.__stride_size) - 1) * self.__stride_size + self.__kernel_size - self.__input_timesteps
            self.__padding_pre = self.__padding_size // 2
            self.__padding_latter = self.__padding_size - self.__padding_pre
            
            self.__input_timesteps += self.__padding_size

        self.__output_size = (self.__input_timesteps - self.__kernel_size) // self.__stride_size + 1

        self.output_size = (self.__output_size, self.__filter_number)

        self.__W = self.__weights_initializer(self.__kernel_size * self.__input_channels, self.__filter_number, (self.__filter_number, self.__input_channels, self.__kernel_size))
        self.__b = np.zeros((self.__filter_number))
        self.__optimizer = optimizer(learning_rate)

    def __img2col(self, img, input_channels):
        col = np.zeros((self.__batch_size, self.__output_size, self.__kernel_size * input_channels))
        for h in range(0, self.__output_size):
            col[:, h, :] = img[:, h*self.__stride_size:h*self.__stride_size+self.__kernel_size, :].reshape(self.__batch_size, -1, order='F')
        
        return col

    def forward(self, X, mode):
        self.__batch_size = X.shape[0]
        self.__input_shape = X.shape

        if self.__padding == 'same':
            X = np.pad(X[:], ((0, 0), (self.__padding_pre, self.__padding_latter), (0, 0)), 'constant')

        self.__col = self.__img2col(X, self.__input_channels)
        return self.__col.dot(self.__W.reshape(self.__filter_number, -1).T) + self.__b[None, None, :]

    def optimize(self, residual):
        g_W = (np.tensordot(residual.reshape(self.__batch_size, self.__filter_number, -1), self.__col, axes=[[0,2], [0, 1]]) / self.__batch_size).reshape(self.__W.shape)
        g_b = np.mean(np.sum(residual, axis=(1, 2)), axis=0)
        
        g_W, g_b = self.__optimizer.optimize([g_W, g_b])
        
        self.__W -= g_W
        self.__b -= g_b

    def backward(self, residual):
        if self.__padding == 'same':
            residual = np.pad(residual[:], ((0, 0), (self.__padding_pre, self.__padding_latter), (0, 0)), 'constant')

        W = self.__W[:, ::-1]

        col = self.__img2col(residual, self.__filter_number)
        return col.dot(W.reshape(self.__input_channels, -1).T)

class Conv2d:
    def __init__(self, filter_number, kernel_shape, stride_size=(1, 1), padding='same',  input_size=0, groups=1, weights_initializer=weights_initializer.he_normal):
        self.__filter_number = filter_number
        self.__input_size = input_size
        self.__kernel_h, self.__kernel_w = kernel_shape
        self.__stride_h, self.__stride_w = stride_size
        self.__padding = padding
        self.__weights_initializer = weights_initializer
        self.__groups = groups

    def init(self, optimizer, learning_rate, input_size=0):
        if self.__input_size == 0:
            self.__input_size = input_size

        self.__input_channels, self.__input_h, self.__input_w = self.__input_size

        if self.__padding == 'same':
            self.__padding_h = (math.ceil(self.__input_h / self.__stride_h) - 1) * self.__stride_h + self.__kernel_h - self.__input_h
            self.__padding_w = (math.ceil(self.__input_w / self.__stride_w) - 1) * self.__stride_w + self.__kernel_w - self.__input_w
            self.__padding_h_up = self.__padding_h // 2
            self.__padding_h_down = self.__padding_h - self.__padding_h_up
            self.__padding_w_left = self.__padding_w // 2
            self.__padding_w_right = self.__padding_w - self.__padding_w_left
            
            self.__input_h += self.__padding_h
            self.__input_w += self.__padding_w

        self.__kernel_size = self.__kernel_h * self.__kernel_w
        self.__output_h = (self.__input_h - self.__kernel_h) // self.__stride_h + 1
        self.__output_w = (self.__input_w - self.__kernel_w) // self.__stride_w + 1
        self.__output_size = self.__output_h * self.__output_w

        self.output_size = (self.__filter_number, self.__output_h, self.__output_w)

        self.__W = self.__weights_initializer(self.__kernel_size * self.__input_channels, self.__filter_number, (self.__filter_number, self.__input_channels, self.__kernel_h, self.__kernel_w))
        self.__b = np.zeros((self.__filter_number))
        self.__optimizer = optimizer(learning_rate)

    def __img2col(self, img, input_channels):
        col = np.zeros((self.__batch_size, self.__output_size, self.__kernel_size * input_channels))
        for h in range(0, self.__output_h):
            for w in range(0, self.__output_w):
                col[:, self.__output_w*h+w, :] = img[:, :, h*self.__stride_h:h*self.__stride_h+self.__kernel_h, w*self.__stride_w:w*self.__stride_w+self.__kernel_w].reshape(self.__batch_size, -1)
        
        return col

    def forward(self, X, mode):
        self.__batch_size = X.shape[0]
        self.__input_shape = X.shape

        if self.__padding == 'same':
            X = np.pad(X[:, :], ((0, 0), (0, 0), (self.__padding_h_up, self.__padding_h_down), (self.__padding_w_left, self.__padding_w_right)), 'constant')

        self.__col = self.__img2col(X, self.__input_channels)
        output = self.__col.dot(self.__W.reshape(self.__filter_number, -1).T)

        return np.transpose(output, axes=(0, 2, 1)).reshape((self.__batch_size, self.__filter_number, self.__output_h, self.__output_w)) + self.__b[None, :, None, None]

    def optimize(self, residual):
        '''
        g_W = np.zeros_like(self.__W)
        for k in range(self.__filter_number):
            tmp = self.__col * residual.reshape(self.__batch_size, self.__filter_number, -1)[:, k, :][:, :, None]
            g_W[k, :, :, :] = np.sum(tmp, axis=(0, 1)).reshape(self.__input_channels, self.__kernel_h, self.__kernel_w) / self.__batch_size
        '''
        g_W = (np.tensordot(residual.reshape(self.__batch_size, self.__filter_number, -1), self.__col, axes=[[0,2], [0, 1]]) / self.__batch_size).reshape(self.__W.shape)
        g_b = np.mean(np.sum(residual, axis=(2, 3)), axis=0)
        
        g_W, g_b = self.__optimizer.optimize([g_W, g_b])
        
        self.__W -= g_W
        self.__b -= g_b

    def backward(self, residual):
        if self.__padding == 'same':
            residual = np.pad(residual[:, :], ((0, 0), (0, 0), (self.__padding_h_up, self.__padding_h_down), (self.__padding_w_left, self.__padding_w_right)), 'constant')

        W = np.transpose(self.__W, axes=(1, 0, 2, 3))
        W = np.rot90(W, k=2, axes=(2, 3))

        col = self.__img2col(residual, self.__filter_number)
        residual = col.dot(W.reshape(self.__input_channels, -1).T)

        return np.transpose(residual, axes=(0, 2, 1)).reshape(self.__input_shape)

class MaxPool:
    def __init__(self, pool_shape, stride_size=None):
        self.__pool_h, self.__pool_w = pool_shape

        if stride_size is None:
            self.__stride_h, self.__stride_w = pool_shape
        else:
            self.__stride_h, self.__stride_w = stride_size

    def init(self, optimizer, learning_rate, input_size=0):
        self.__input_channels, self.__input_h, self.__input_w = input_size
        self.__output_h = (self.__input_h - self.__pool_h) // self.__stride_h + 1
        self.__output_w = (self.__input_w - self.__pool_w) // self.__stride_w + 1
        self.output_size = (self.__input_channels, self.__output_h, self.__output_w)

    def forward(self, X, mode):        
        self.__batch_size = X.shape[0]

        output = np.zeros((self.__batch_size, self.__input_channels, self.__output_h, self.__output_w))
        self.__output_index = np.zeros_like(X)
        for h in range(0, self.__output_h):
            for w in range(0, self.__output_w):
                output_index = np.argmax(X[:, :, h*self.__stride_h:h*self.__stride_h+self.__pool_h, w*self.__stride_w:w*self.__stride_w+self.__pool_w].reshape((self.__batch_size, self.__input_channels, -1)), axis=2)
                output_index_h, output_index_w = divmod(output_index, self.__pool_w)
                for i in range(self.__batch_size):
                    for j in range(self.__input_channels):
                        self.__output_index[i, j, h*self.__stride_h+output_index_h[i, j], w*self.__stride_w+output_index_w[i, j]] = 1
                output[:, :, h, w] = np.max(X[:, :, h*self.__stride_h:h*self.__stride_h+self.__pool_h, w*self.__stride_w:w*self.__stride_w+self.__pool_w], axis=(2, 3))
        
        return output
        
    def backward(self, residual):
        if self.__stride_h == self.__pool_h and self.__stride_w == self.__pool_w:
            layer_residual = np.repeat(residual, repeats=self.__pool_h, axis=2)
            layer_residual = np.repeat(layer_residual, repeats=self.__pool_w, axis=3)
        else:
            layer_residual = np.zeros((self.__batch_size, self.__input_channels, self.__input_h, self.__input_w))
            for h in range(0, self.__output_h):
                for w in range(0, self.__output_w):
                    layer_residual[:, :, h*self.__stride_h:h*self.__stride_h+self.__pool_h, w*self.__stride_w:w*self.__stride_w+self.__pool_w] += residual[:, :, h, w][:, :, None, None]

        return layer_residual * self.__output_index

class MeanPool:
    def __init__(self, pool_shape, stride_size=None):
        self.__pool_h, self.__pool_w = pool_shape

        if stride_size is None:
            self.__stride_h, self.__stride_w = pool_shape
        else:
            self.__stride_h, self.__stride_w = stride_size

    def init(self, optimizer, learning_rate, input_size=0):
        self.__input_channels, self.__input_h, self.__input_w = input_size
        self.__output_h = (self.__input_h - self.__pool_h) // self.__stride_h + 1
        self.__output_w = (self.__input_w - self.__pool_w) // self.__stride_w + 1
        self.output_size = (self.__input_channels, self.__output_h, self.__output_w)

    def forward(self, X, mode):
        self.__batch_size = X.shape[0]
        
        M, N = X.shape[2], X.shape[3]
        K = self.__pool_h
        L = self.__pool_w

        MK = M // K
        NL = N // L

        return X[:, :, :MK*K, :NL*L].reshape(X.shape[0], X.shape[1], MK, K, NL, L).mean(axis=(3, 5))
        '''
        output = np.zeros((self.__batch_size, self.__input_channels, self.__output_h, self.__output_w))
        for h in range(0, self.__output_h):
            for w in range(0, self.__output_w):
                output[:, :, h, w] = np.mean(X[:, :, h*self.__stride_h:h*self.__stride_h+self.__pool_h, w*self.__stride_w:w*self.__stride_w+self.__pool_w], axis=(2, 3))

        return output
        '''

    def backward(self, residual):
        residual /= self.__pool_h * self.__pool_w

        if self.__stride_h == self.__pool_h and self.__stride_w == self.__pool_w:
            layer_residual = np.repeat(residual, repeats=self.__pool_h, axis=2)
            layer_residual = np.repeat(layer_residual, repeats=self.__pool_w, axis=3)
        else:
            layer_residual = np.zeros((self.__batch_size, self.__input_channels, self.__input_h, self.__input_w))
            for h in range(0, self.__output_h):
                for w in range(0, self.__output_w):
                    layer_residual[:, :, h*self.__stride_h:h*self.__stride_h+self.__pool_h, w*self.__stride_w:w*self.__stride_w+self.__pool_w] += residual[:, :, h, w][:, :, None, None]

        return layer_residual

class Flatten:
    def init(self, optimizer, learning_rate, input_size=0):
        self.output_size = reduce(lambda i, j : i * j, input_size)

    def forward(self, X, mode):
        self.__input_shape = X.shape
        return X.reshape(self.__input_shape[0], -1)

    def backward(self, residual):
        return residual.reshape(self.__input_shape)

class Dropout:
    def __init__(self, p=0):
        self.__p = p

    def init(self, optimizer, learning_rate, input_size=0):
        self.output_size = input_size
    
    def forward(self, X, mode):
        if self.__p > 0 and mode == 'fit':
            self.__dropout_index = random.sample(range(self.output_size), int(self.output_size * self.__p))
            X[:, self.__dropout_index] = 0
            return X / (1 - self.__p)
        else:
            return X

    def backward(self, residual):
        if self.__p > 0:
            residual[:, self.__dropout_index] = 0

        return residual / (1 - self.__p)

class BatchNormalization:
    def init(self, optimizer, learning_rate, input_size=0):
        self.output_size = input_size
        self.__gamma = np.ones(self.output_size)
        self.__beta = np.zeros(self.output_size)
        self.__optimizer = optimizer(learning_rate)
        self.__predict_mean = 0
        self.__predict_std = 0

    def forward(self, X, mode):
        momentum = 0.999
        
        self.__X = X

        if mode == 'fit':
            self.__mean = np.mean(self.__X, axis=0)
            self.__predict_mean = momentum * self.__predict_mean + (1 - momentum) * self.__mean

            self.__std = np.std(self.__X, axis=0)
            self.__predict_std = momentum * self.__predict_std + (1 - momentum) * self.__std

            self.__X_hat = (self.__X - self.__mean) / (self.__std + 1e-8)
        elif mode == 'predict':
            self.__X_hat = (self.__X - self.__predict_mean) / (self.__predict_std + 1e-8)
        
        return self.__gamma * self.__X_hat + self.__beta

    def backward(self, residual):
        d_X_hat = residual * self.__gamma
        
        d_var =  np.sum(-d_X_hat * (self.__X - self.__mean) * ((self.__std + 1e-8) ** -3) / 2, axis=0)

        d_mean = np.sum(-d_X_hat / (self.__std + 1e-8), axis=0) - 2 * d_var * np.mean(self.__X - self.__mean)

        batch_size = self.__X.shape[0]
        return d_X_hat / (self.__std + 1e-8) + d_var * 2 * (self.__X - self.__mean) / batch_size + d_mean / batch_size

    def optimize(self, residual):
        g_gamma = np.mean(residual * self.__X_hat, axis=0)
        g_beta = np.mean(residual, axis=0)

        g_gamma, g_beta = self.__optimizer.optimize([g_gamma, g_beta])

        self.__gamma -= g_gamma
        self.__beta -= g_beta

class Rnn:
    class __RnnCell:
        def __init__(self, input_size, output_size, optimizer):
            self.__U = weights_initializer.xavier_normal(output_size, output_size, (output_size, output_size))
            self.__U_gradient = 0
            self.__W = weights_initializer.xavier_normal(input_size, output_size, (input_size, output_size))
            self.__W_gradient = 0
            self.__b = np.zeros(output_size)
            self.__b_gradient = 0

            self.__V = weights_initializer.xavier_normal(output_size, output_size, (output_size, output_size))
            self.__V_gradient = 0
            self.__c = np.zeros(output_size)
            self.__c_gradient = 0

            self.__optimizer = optimizer

        def forward(self, X, h_pre):
            self.__X = X
            self.__h_pre = h_pre

            self.__h = np.tanh(self.__X.dot(self.__W) + self.__h_pre.dot(self.__U) + self.__b)
            Z = self.__h.dot(self.__V) + self.__c

            return self.__h, Z

        def backward(self, residual):
            batch_size = residual.shape[0]

            self.__V_gradient += self.__h.T.dot(residual) / batch_size
            self.__c_gradient += np.mean(residual, axis=0)

            residual = residual.dot(self.__V.T)
            residual *= (1 - self.__h ** 2)

            self.__W_gradient += self.__X.T.dot(residual) / batch_size
            self.__b_gradient += np.mean(residual, axis=0)
            self.__U_gradient += self.__h_pre.T.dot(residual) / batch_size

            return residual.dot(self.__W.T), residual.dot(self.__U.T)

        def optimize(self):
            g_U, g_W, g_b, g_V, g_c = self.__optimizer.optimize([self.__U_gradient, self.__W_gradient, self.__b_gradient, self.__V_gradient, self.__c_gradient])

            self.__U -= g_U
            self.__W -= g_W
            self.__b -= g_b
            self.__V -= g_V
            self.__c -= g_c
            self.__U_gradient = 0
            self.__W_gradient = 0
            self.__b_gradient = 0
            self.__V_gradient = 0
            self.__c_gradient = 0
        
    def __init__(self, time_steps, output_size, activivation, layer_size, input_size=0):
        self.output_size = output_size
        self.__input_size = input_size
        self.__time_steps = time_steps
        self.__layer_size = layer_size
        self.__activations = np.array([activivation() for i in range(self.__time_steps * self.__layer_size)]).reshape((self.__layer_size, self.__time_steps))

    def init(self, optimizer, learning_rate, input_size=0):
        if self.__input_size == 0:
            self.__input_size = input_size

        self.__rnn_cells = []
        self.__rnn_cells.append(self.__RnnCell(self.__input_size, self.output_size, optimizer(learning_rate)))
        for i in range(1, self.__layer_size):
            self.__rnn_cells.append(self.__RnnCell(self.output_size, self.output_size, optimizer(learning_rate)))

    def forward(self, X, mode):
        batch_size = X.shape[0]

        h = np.zeros((batch_size, self.__time_steps, self.__layer_size, self.output_size))

        for i in range(self.__time_steps):
            y = X[:, i]
            for j in range(self.__layer_size):
                if i == 0:
                    h_pre = np.zeros((batch_size, self.output_size))
                else:
                    h_pre = h[:, i - 1, j]

                h[:, i, j], Z = self.__rnn_cells[j].forward(y, h_pre)
                y = self.__activations[j, i].forward(Z, mode)

        return y

    def optimize(self, residual):
        residual_X = residual
        residual_h = [0 for i in range(self.__layer_size)]
        for i in range(self.__time_steps - 1, -1, -1):
            for j in range(self.__layer_size - 1, -1, -1):
                residual_tmp = self.__activations[j, i].backward(residual_X + residual_h[j])
                residual_X, residual_h[j] = self.__rnn_cells[j].backward(residual_tmp)

            residual_X = 0

        for i in range(self.__layer_size):
            self.__rnn_cells[i].optimize()

class Dense:
    def __init__(self, output_size, input_size=0, regularizer=regularizer.Regularizer(0), weights_initializer=weights_initializer.he_normal):
        self.output_size = output_size
        self.__input_size = input_size
        self.__regularizer = regularizer
        self.__weights_initializer = weights_initializer

    def init(self, optimizer, learning_rate, input_size=0):
        if self.__input_size == 0:
            self.__input_size = input_size

        self.__W = self.__weights_initializer(self.__input_size, self.output_size, (self.__input_size, self.output_size))
        self.__b = np.zeros(self.output_size)

        self.__optimizer = optimizer(learning_rate)

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

class Relu:
    def init(self, optimizer, learning_rate, input_size=0):
        self.output_size = input_size
        
    def forward(self, X, mode):
        self.__X = X
        return np.maximum(self.__X, 0)

    def backward(self, residual):
        return (self.__X > 0) * residual

class Selu:
    def __init__(self):
        self.__alpha = 1.6732632423543772848170429916717
        self.__scale = 1.0507009873554804934193349852946
    
    def init(self, optimizer, learning_rate, input_size=0):
        self.output_size = input_size

    def forward(self, X, mode):
        self.__X = X
        return self.__scale * np.where(self.__X > 0.0, self.__X, self.__alpha * (np.exp(self.__X) - 1))

    def backward(self, residual):
        return self.__scale * np.where(self.__X > 0.0, 1, self.__alpha * (np.exp(self.__X) - 1)) * residual

class Tanh:
    def init(self, optimizer, learning_rate, input_size=0):
        self.output_size = input_size

    def forward(self, X, mode):
        self.__output = np.tanh(X)
        return self.__output

    def backward(self, residual):
        return (1 - self.__output ** 2) * residual

class Sigmoid:
    def forward(self, X, mode):
        return 1 / (1 + np.exp(-X))

    def backward(self, residual):
        return residual

class Softmax:
    def forward(self, X, mode):
        X_max = np.max(X, axis=1, keepdims=True)
        X_exp = np.exp(X - X_max)
        return X_exp / np.sum(X_exp, axis=1, keepdims=True)

    def backward(self, residual):
        return residual

class Nnet:
    def __init__(self, loss, optimizer, learning_rate, debug=True):
        self.__debug = debug
        self.__layers = []
        self.__loss = loss
        self.__optimizer = optimizer
        self.__learning_rate = learning_rate

    def __get_residual(self, h, y):
        if self.__loss == 'binary_crossentropy' or self.__loss == 'mse' or self.__loss == 'categorical_crossentropy':
            return h - y
        elif self.__loss == 'categorical_hinge':
            batch_size = y.shape[0]
            residual = np.zeros_like(y)
            correct_index = np.argmax(y, axis=1)
            residual[np.flatnonzero(h - h[range(batch_size), correct_index].reshape((-1, 1)) + 1 > 0)] = 1
            residual[range(batch_size), correct_index] = 0
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
            loss = -np.mean(np.sum(y * np.log(h) + (1 - y) * np.log(1 - h), axis=1))
        elif self.__loss == 'mse':
            loss = np.mean((h - y) ** 2)
        elif self.__loss == 'categorical_crossentropy':
            loss = -np.mean(np.sum(y * np.log(h + 1e-8), axis=1))
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
                layer.init(self.__optimizer, self.__learning_rate, self.__layers[-1].output_size)
            else:
                layer.init(self.__optimizer, self.__learning_rate)

        self.__layers.append(layer)

    def fit(self, X, y, batch_size, epochs):
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
                    
            threading.Thread(target=self.__log, args=(_, loss[-1], timeit.default_timer() - start_time, np.mean(accuracy))).start()

        if self.__debug:
            self.__draw_figure(loss, accuracy)

    def predict(self, X, classes=None):
        if self.__loss == 'categorical_crossentropy' or self.__loss == 'categorical_hinge':
            return classes[np.argmax(self.score(X), axis=1)].reshape(-1, 1)
        elif self.__loss == 'binary_crossentropy':
            return np.around(self.score(X)).reshape(-1, 1)
        elif self.__loss == 'mse':
            return self.score(X).reshape(-1, 1)

    def score(self, X):
        return self.__foreward(X, 'predict')