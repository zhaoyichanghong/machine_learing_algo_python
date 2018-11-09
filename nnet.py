import numpy as np
import matplotlib.pyplot as plt
import timeit
import metrics
import optimizer
import preprocess
import threading
from sklearn.utils import shuffle
import random
import math
from functools import reduce

class layer:
    def init(self, optimizer, learning_rate, input_number=0):
        self.unit_number = input_number

    def forward(self, X, mode):
        pass

    def backward(self, y, residual):
        return residual

    def optimize (self, y, residual):
        pass

class conv2d(layer):
    def __init__(self, filter_number, kernel_shape, stride_size=(1, 1), padding='same',  input_number=0):
        self.__filter_number = filter_number
        self.__input_number = input_number
        self.__kernel_h, self.__kernel_w = kernel_shape
        self.__stride_h, self.__stride_w = stride_size
        self.__padding = padding

    def init(self, optimizer, learning_rate, input_number=0):
        if self.__input_number == 0:
            self.__input_number = input_number

        self.__input_channels, self.__input_h, self.__input_w = self.__input_number

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

        self.unit_number = (self.__filter_number, self.__output_h, self.__output_w)

        self.__W = np.random.normal(scale=np.sqrt(4 / (self.__kernel_size + self.__filter_number)), size=(self.__filter_number, self.__input_channels, self.__kernel_h, self.__kernel_w))
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

    def optimize(self, y, residual):
        '''
        g_W = np.zeros_like(self.__W)
        for k in range(self.__filter_number):
            tmp = self.__col * residual.reshape(self.__batch_size, self.__filter_number, -1)[:, k, :][:, :, None]
            g_W[k, :, :, :] = np.sum(tmp, axis=(0, 1)).reshape(self.__input_channels, self.__kernel_h, self.__kernel_w) / self.__batch_size
        '''
        g_W = (np.tensordot(residual.reshape(self.__batch_size, self.__filter_number, -1), self.__col, axes=[[0,2], [0, 1]]) / self.__batch_size).reshape(self.__W.shape)
        g_b = np.mean(np.sum(residual, axis=(2, 3)), axis=0)
        
        g_W, g_b = self.__optimizer.optimize(g_W, g_b)
        
        self.__W -= g_W
        self.__b -= g_b

    def backward(self, y, residual):
        if self.__padding == 'same':
            residual = np.pad(residual[:, :], ((0, 0), (0, 0), (self.__padding_h_up, self.__padding_h_down), (self.__padding_w_left, self.__padding_w_right)), 'constant')

        W = np.transpose(self.__W, axes=(1, 0, 2, 3))
        W = np.rot90(W, k=2, axes=(2, 3))

        col = self.__img2col(residual, self.__filter_number)
        residual = col.dot(W.reshape(self.__input_channels, -1).T)

        return np.transpose(residual, axes=(0, 2, 1)).reshape(self.__input_shape)

class flatten(layer):
    def init(self, optimizer, learning_rate, input_number=0):
        self.unit_number = reduce(lambda i, j : i * j, input_number)

    def forward(self, X, mode):
        self.__input_shape = X.shape
        return X.reshape(self.__input_shape[0], -1)

    def backward(self, y, residual):
        return residual.reshape(self.__input_shape)

class dropout(layer):
    def __init__(self, p=0):
        self.__p = p
    
    def forward(self, X, mode):
        if self.__p > 0 and mode == 'fit':
            self.__dropout_index = random.sample(range(self.unit_number), int(self.unit_number * self.__p))
            X[:, self.__dropout_index] = 0
            return X / (1 - self.__p)
        else:
            return X

    def backward(self, y, residual):
        if self.__p > 0:
            residual[:, self.__dropout_index] = 0

        return residual / (1 - self.__p)

class dense(layer):
    def __init__(self, unit_number, input_number=0):
        self.unit_number = unit_number
        self.__input_number = input_number

    def init(self, optimizer, learning_rate, input_number=0):
        if self.__input_number == 0:
            self.__input_number = input_number

        self.__W = np.random.normal(scale=np.sqrt(4 / (self.__input_number + self.unit_number)), size=(self.__input_number, self.unit_number))
        self.__b = np.zeros(self.unit_number)

        self.__optimizer = optimizer(learning_rate)

    def forward(self, X, mode):
        self.__X = X
        return self.__X.dot(self.__W) + self.__b

    def backward(self, y, residual):
        return residual.dot(self.__W.T)

    def optimize(self, y, residual):
        batch_size = self.__X.shape[0]
        g_W = self.__X.T.dot(residual) / batch_size
        g_b = np.mean(residual, axis=0)

        g_W, g_b = self.__optimizer.optimize(g_W, g_b)

        self.__W -= g_W
        self.__b -= g_b

class relu(layer):
    def forward(self, X, mode):
        self.__X = X
        return np.maximum(self.__X, 0)

    def backward(self, y, residual):
        return (self.__X > 0) * residual

class selu(layer):
    def __init__(self):
        self.__alpha = 1.6732632423543772848170429916717
        self.__scale = 1.0507009873554804934193349852946

    def forward(self, X, mode):
        self.__X = X
        return self.__scale * np.where(self.__X > 0.0, self.__X, self.__alpha * (np.exp(self.__X) - 1))

    def backward(self, y, residual):
        return self.__scale * np.where(self.__X > 0.0, 1, self.__alpha * (np.exp(self.__X) - 1)) * residual

class tanh(layer):
    def forward(self, X, mode):
        self.__output = np.tanh(X)
        return self.__output

    def backward(self, y, residual):
        return (1 - np.power(self.__output, 2)) * residual

class sigmoid(layer):
    def forward(self, X, mode):
        return 1 / (1 + np.exp(-X))

class softmax(layer):
    def forward(self, X, mode):
        X_max = np.max(X, axis=1, keepdims=True)
        X_exp = np.exp(X - X_max)
        return X_exp / np.sum(X_exp, axis=1, keepdims=True)

class nnet:
    def __init__(self, loss, optimizer, learning_rate):
        self.layers = []
        self.__loss = loss
        self.__optimizer = optimizer
        self.__learning_rate = learning_rate

    def __get_residual(self, h, y):
        y = y.reshape(h.shape)

        if self.__loss == 'binary_crossentropy' or self.__loss == 'mse' or self.__loss == 'categorical_crossentropy':
            return h - y
        elif self.__loss == 'categorical_hinge':
            batch_size, class_number = y.shape

            residual = np.zeros_like(y)
            correct_index = np.argmax(y, axis=1)
            for i in range(batch_size):
                for j in range(class_number):
                    if j != correct_index[i]:
                        if h[i, j] - h[i, correct_index[i]] + 1 > 0:
                            residual[i, j] = 1
                            residual[i, correct_index[i]] -= 1

            return residual
           
    def __get_accuracy(self, h, y):
        if self.__loss == 'categorical_crossentropy' or self.__loss == 'categorical_hinge':
            return np.mean(np.argmax(h, axis=1) == np.argmax(y, axis=1))
        elif self.__loss == 'binary_crossentropy':
            return np.mean(np.around(h) == y)
        else:
            return 0

    def __get_loss(self, h, y):
        y = y.reshape(h.shape)

        if self.__loss == 'binary_crossentropy':
            loss = -np.mean(np.sum(y * np.log(h) + (1 - y) * np.log(1 - h), axis=1))
        elif self.__loss == 'mse':
            loss = np.mean((h - y) ** 2)
        elif self.__loss == 'categorical_crossentropy':
            loss = -np.mean(np.sum(y * np.log(h + 1e-8), axis=1))
        elif self.__loss == 'categorical_hinge':
            batch_size, class_number = y.shape

            loss = np.zeros(batch_size)
            correct_index = np.argmax(y, axis=1)
            for i in range(batch_size):
                for j in range(class_number):
                    if j != correct_index[i]:
                        loss[i] += max(0, h[i, j] - h[i, correct_index[i]] + 1)

            loss = np.mean(loss)

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
        for layer in self.layers:
            X = layer.forward(X, mode)

        return X

    def __backward(self, y, residual):
        for layer in reversed(self.layers):
            residual_back = residual
            if layer != self.layers[0]:
                residual = layer.backward(y, residual)
            if hasattr(layer, 'optimize'):
                layer.optimize(y, residual_back)

    def add(self, layer):
        if len(self.layers) > 0:
            layer.init(self.__optimizer, self.__learning_rate, self.layers[-1].unit_number)
        else:
            layer.init(self.__optimizer, self.__learning_rate)

        self.layers.append(layer)

    def fit(self, X, y, batch_size, epochs):
        if self.__loss == 'categorical_crossentropy' or self.__loss == 'categorical_hinge':
            encoder = preprocess.one_hot()
            y = encoder.fit_transform(y)
            self.__classes = np.array(encoder.classes)

        data_number = X.shape[0]
        epoch = (data_number + batch_size - 1) // batch_size

        loss = []
        accuracy = []
        for _ in range(epochs):
            start_time = timeit.default_timer()

            X_epoch, y_epoch = shuffle(X, y)
            for i in range(epoch):
                X_batch = X_epoch[batch_size*i : min(batch_size*(i+1), data_number)]
                y_batch = y_epoch[batch_size*i : min(batch_size*(i+1), data_number)]
                    
                h = self.__foreward(X_batch)
                residual = self.__get_residual(h, y_batch)
                self.__backward(y_batch, residual)

                loss.append(self.__get_loss(h, y_batch))
                accuracy.append(self.__get_accuracy(h, y_batch))
                    
            threading.Thread(target=self.__log, args=(_, loss[-1], timeit.default_timer() - start_time, np.mean(accuracy))).start()

        self.__draw_figure(loss, accuracy)

    def predict(self, X):
        if self.__loss == 'categorical_crossentropy' or self.__loss == 'categorical_hinge':
            return self.__classes[np.argmax(self.score(X), axis=1)].reshape(-1, 1)
        elif self.__loss == 'binary_crossentropy':
            return np.around(self.score(X)).reshape(-1, 1)
        elif self.__loss == 'mse':
            return self.score(X).reshape(-1, 1)

    def score(self, X):
        self.__mode = 'predict'
        return self.__foreward(X, 'predict')