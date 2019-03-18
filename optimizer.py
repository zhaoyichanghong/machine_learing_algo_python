import numpy as np

class GradientDescent:
    def __init__(self, learning_rate):
        '''
        Parameters
        ----------
        learning_rate : Learning rate
        '''
        self.__learning_rate = learning_rate

    def optimize(self, vars):
        '''
        Parameters
        ----------
        vars : Parameters to be optimized

        Returns
        -------
        vars : Parameters optimized
        '''
        return [self.__learning_rate * vars[i] for i in range(len(vars))]

class Momentum:
    def __init__(self, learning_rate):
        '''
        Parameters
        ----------
        learning_rate : Learning rate
        '''
        self.__learning_rate = learning_rate
        self.__alpha = 0.9

    def optimize(self, vars):
        '''
        Parameters
        ----------
        vars : Parameters to be optimized

        Returns
        -------
        vars : Parameters optimized
        '''
        n_vars = len(vars)

        if not hasattr(self, '_Momentum__v'):
            self.__v = [0] * n_vars

        self.__v = [(1 - self.__alpha) * vars[i] + self.__alpha * self.__v[i] for i in range(n_vars)]

        return self.__learning_rate * np.array(self.__v)

class Nesterov:
    def __init__(self, learning_rate):
        '''
        Parameters
        ----------
        learning_rate : Learning rate
        '''
        self.__learning_rate = learning_rate
        self.__alpha = 0.9

    def optimize(self, vars):
        '''
        Parameters
        ----------
        vars : Parameters to be optimized

        Returns
        -------
        vars : Parameters optimized
        '''
        n_vars = len(vars)

        if not hasattr(self, '_Nesterov__v'):
            self.__v = [0] * n_vars
        
        old_v = self.__v
        self.__v = [self.__alpha * self.__v[i] - self.__learning_rate * vars[i] for i in range(n_vars)]

        return self.__alpha * np.array(old_v) - (1 + self.__alpha) * np.array(self.__v)

class Adagrad:
    def __init__(self, learning_rate):
        '''
        Parameters
        ----------
        learning_rate : Learning rate
        '''
        self.__learning_rate = learning_rate

    def optimize(self, vars):
        '''
        Parameters
        ----------
        vars : Parameters to be optimized

        Returns
        -------
        vars : Parameters optimized
        '''
        n_vars = len(vars)

        if not hasattr(self, '_Adagrad__r'):
            self.__r = [0] * n_vars

        v = [0] * n_vars
        for i in range(n_vars):
            self.__r[i] += vars[i] ** 2
            v[i] = self.__learning_rate * vars[i] / (np.sqrt(self.__r[i]) + 1e-8)

        return v

class Rmsprop:
    def __init__(self, learning_rate):
        '''
        Parameters
        ----------
        learning_rate : Learning rate
        '''
        self.__learning_rate = learning_rate
        self.__alpha = 0.9

    def optimize(self, vars):
        '''
        Parameters
        ----------
        vars : Parameters to be optimized

        Returns
        -------
        vars : Parameters optimized
        '''
        n_vars = len(vars)

        if not hasattr(self, '_Rmsprop__r'):
            self.__r = [0] * n_vars

        v = [0] * n_vars
        for i in range(n_vars):
            self.__r[i] = self.__alpha * self.__r[i] + (1 - self.__alpha) * vars[i] ** 2
            v[i] = self.__learning_rate * vars[i] / (np.sqrt(self.__r[i]) + 1e-8)

        return v

class Adam:
    def __init__(self, learning_rate):
        '''
        Parameters
        ----------
        learning_rate : Learning rate
        '''
        self.__learning_rate = learning_rate
        self.__t = 0
        self.__alpha = 0.9
        self.__alpha2 = 0.999

    def optimize(self, vars):
        '''
        Parameters
        ----------
        vars : Parameters to be optimized

        Returns
        -------
        vars : Parameters optimized
        '''
        n_vars = len(vars)

        if not hasattr(self, '_Adam__r'):
            self.__s = [0] * n_vars
            self.__r = [0] * n_vars

        self.__t += 1

        v = [0] * n_vars
        for i in range(n_vars):
            self.__s[i] = self.__alpha * self.__s[i] + (1 - self.__alpha) * vars[i]
            s_hat = self.__s[i] / (1 - self.__alpha ** self.__t)
            self.__r[i] = self.__alpha2 * self.__r[i] + (1 - self.__alpha2) * np.power(vars[i], 2)
            r_hat = self.__r[i] / (1 - self.__alpha2 ** self.__t)
            v[i] = self.__learning_rate * s_hat / (1e-8 + np.sqrt(r_hat))

        return  v