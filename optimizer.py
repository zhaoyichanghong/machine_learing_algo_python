import numpy as np

class GradientDescent:
    def __init__(self, learning_rate):
        self.__learning_rate = learning_rate
        self.__first_run = True

    def optimize(self, vars):
        if self.__first_run:
            self.__vars_number = len(vars)
            self.__first_run = False

        v = [0] * self.__vars_number
        for i in range(self.__vars_number):
            v[i] = self.__learning_rate * vars[i]

        return v

class CgPrp:
    def __init__(self, learning_rate):
        self.__learning_rate = learning_rate
        self.__first_run = True

    def optimize(self, vars):
        if self.__first_run:
            self.__vars_number = len(vars)
            self.__g = [np.zeros((1, 1))] * self.__vars_number
            self.__d = [0] * self.__vars_number
            self.__first_run = False

        v = [0] * self.__vars_number
        for i in range(self.__vars_number):
            beta = np.linalg.norm(vars[i], axis=0) ** 2 / (np.linalg.norm(self.__g[i], axis=0) ** 2 + 1e-8)
            d = vars[i] - beta * self.__d[i]
            self.__d[i] = -d
            self.__g[i] = vars[i]
            v[i] = self.__learning_rate * d.reshape(vars[i].shape)

        return v

class CgFr:
    def __init__(self, learning_rate):
        self.__learning_rate = learning_rate
        self.__first_run = True

    def optimize(self, vars):
        if self.__first_run:
            self.__vars_number = len(vars)
            self.__g = [np.zeros((1, 1))] * self.__vars_number
            self.__d = [0] * self.__vars_number
            self.__first_run = False

        v = [0] * self.__vars_number
        for i in range(self.__vars_number):
            beta = np.sum(vars[i] * (vars[i] - self.__g[i]), axis=0) / (np.linalg.norm(self.__g[i], axis=0) ** 2 + 1e-8)
            d = vars[i] - beta * self.__d[i]
            self.__d[i] = -d
            self.__g[i] = vars[i]
            v[i] = self.__learning_rate * d.reshape(vars[i].shape)

        return v

class Momentum:
    def __init__(self, learning_rate):
        self.__learning_rate = learning_rate
        self.__first_run = True
        self.__alpha = 0.9

    def optimize(self, vars):
        if self.__first_run:
            self.__vars_number = len(vars)
            self.__v = [0] * self.__vars_number
            self.__first_run = False

        v = [0] * self.__vars_number
        for i in range(self.__vars_number):
            self.__v[i] = (1 - self.__alpha) * vars[i] + self.__alpha * self.__v[i]
            v[i] = self.__learning_rate * self.__v[i]

        return v

class Adagrad:
    def __init__(self, learning_rate):
        self.__learning_rate = learning_rate
        self.__first_run = True

    def optimize(self, vars):
        if self.__first_run:
            self.__vars_number = len(vars)
            self.__r = [0] * self.__vars_number
            self.__first_run = False

        v = [0] * self.__vars_number
        for i in range(self.__vars_number):
            self.__r[i] += vars[i] ** 2
            v[i] = self.__learning_rate * vars[i] / (np.sqrt(self.__r[i]) + 1e-8)

        return v

class Rmsprop:
    def __init__(self, learning_rate):
        self.__learning_rate = learning_rate
        self.__first_run = True
        self.__alpha = 0.9

    def optimize(self, vars):
        if self.__first_run:
            self.__vars_number = len(vars)
            self.__r = [0] * self.__vars_number
            self.__first_run = False

        v = [0] * self.__vars_number
        for i in range(self.__vars_number):
            self.__r[i] = self.__alpha * self.__r[i] + (1 - self.__alpha) * vars[i] ** 2
            v[i] = self.__learning_rate * vars[i] / (np.sqrt(self.__r[i]) + 1e-8)

        return v

class Adam:
    def __init__(self, learning_rate):
        self.__learning_rate = learning_rate
        self.__first_run = True
        self.__t = 0
        self.__alpha = 0.9
        self.__alpha2 = 0.999

    def optimize(self, vars):
        if self.__first_run:
            self.__vars_number = len(vars)
            self.__s = [0] * self.__vars_number
            self.__r = [0] * self.__vars_number
            self.__first_run = False

        self.__t += 1

        v = [0] * self.__vars_number
        for i in range(self.__vars_number):
            self.__s[i] = self.__alpha * self.__s[i] + (1 - self.__alpha) * vars[i]
            s_hat = self.__s[i] / (1 - self.__alpha ** self.__t)
            self.__r[i] = self.__alpha2 * self.__r[i] + (1 - self.__alpha2) * np.power(vars[i], 2)
            r_hat = self.__r[i] / (1 - self.__alpha2 ** self.__t)
            v[i] = self.__learning_rate * s_hat / (1e-8 + np.sqrt(r_hat))

        return  v