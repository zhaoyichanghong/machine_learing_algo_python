import numpy as np
import matplotlib.pyplot as plt
import timeit
import metrics
import optimizer
import preprocess
import threading
from sklearn.utils import shuffle

class layer:
    def init(self, optimizer, learning_rate, input_number=0):
        self.unit_number = input_number

    def forward(self, X):
        pass

    def backward(self, y, residual):
        return residual

    def optimize (self, y, residual):
        pass

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

    def forward(self, X):
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
    def forward(self, X):
        self.__X = X
        return np.maximum(self.__X, 0)

    def backward(self, y, residual):
        return (self.__X > 0) * residual

class selu(layer):
    def __init__(self):
        self.__alpha = 1.6732632423543772848170429916717
        self.__scale = 1.0507009873554804934193349852946

    def forward(self, X):
        self.__X = X
        return self.__scale * np.where(self.__X > 0.0, self.__X, self.__alpha * (np.exp(self.__X) - 1))

    def backward(self, y, residual):
        return self.__scale * np.where(self.__X > 0.0, 1, self.__alpha * (np.exp(self.__X) - 1)) * residual

class tanh(layer):
    def forward(self, X):
        self.__output = np.tanh(X)
        return self.__output

    def backward(self, y, residual):
        return (1 - np.power(self.__output, 2)) * residual

class sigmoid(layer):
    def forward(self, X):
        return 1 / (1 + np.exp(-X))

class softmax(layer):
    def forward(self, X):
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
            loss = -np.mean(np.sum(y * np.log(h), axis=1))
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

    def __foreward(self, X):
        for layer in self.layers:
            X = layer.forward(X)

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
        return self.__foreward(X)