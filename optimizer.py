import numpy as np

class optimizer:
    def __init__(self, learning_rate=0):
        self.learning_rate = learning_rate

    def optimize(self, g_w, g_b):
        pass

class gradient_descent(optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)

    def optimize(self, g_w, g_b):
        return self.learning_rate * g_w, self.learning_rate * g_b

class cg_prp(optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)
        self.__g_w = np.zeros((1, 1))
        self.__d_w = np.zeros((1, 1))
        self.__g_b = np.zeros((1, 1))
        self.__d_b = np.zeros((1, 1))

    def optimize(self, g_w, g_b):
        beta = np.linalg.norm(g_w, axis=0) ** 2 / (np.linalg.norm(self.__g_w, axis=0) ** 2 + 1e-8)
        d_w = g_w - beta * self.__d_w
        self.__d_w = -d_w
        self.__g_w = g_w

        if not hasattr(g_b, "__len__"):
            g_b = np.full((1, 1), g_b)

        beta = np.linalg.norm(g_b, axis=0) ** 2 / (np.linalg.norm(self.__g_b, axis=0) ** 2 + 1e-8)
        d_b = g_b - beta * self.__d_b
        self.__d_b = -d_b
        self.__g_b = g_b

        return self.learning_rate * d_w.reshape(g_w.shape), self.learning_rate * d_b.reshape(g_b.shape)

class cg_fr(optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)
        self.__g_w = np.zeros((1, 1))
        self.__d_w = np.zeros((1, 1))
        self.__g_b = np.zeros((1, 1))
        self.__d_b = np.zeros((1, 1))

    def optimize(self, g_w, g_b):
        beta = np.sum(g_w * (g_w - self.__g_w), axis=0) / (np.linalg.norm(self.__g_w, axis=0) ** 2 + 1e-8)
        d_w = g_w - beta * self.__d_w
        self.__d_w = -d_w
        self.__g_w = g_w

        if not hasattr(g_b, "__len__"):
            g_b = np.full((1, 1), g_b)

        beta = np.sum(g_b * (g_b - self.__g_b), axis=0) / (np.linalg.norm(self.__g_b, axis=0) ** 2 + 1e-8)
        d_b = g_b - beta * self.__d_b
        self.__d_b = -d_b
        self.__g_b = g_b

        return self.learning_rate * d_w.reshape(g_w.shape), self.learning_rate * d_b.reshape(g_b.shape)

class momentum(optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)
        self.__alpha = 0.9
        self.__v_w = 0
        self.__v_b = 0

    def optimize(self, g_w, g_b):
        self.__v_w = (1 - self.__alpha) * g_w + self.__alpha * self.__v_w
        self.__v_b = (1 - self.__alpha) * g_b + self.__alpha * self.__v_b

        return self.learning_rate * self.__v_w, self.learning_rate * self.__v_b

class adagrad(optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)
        self.__r_w = 0
        self.__r_b = 0

    def optimize(self, g_w, g_b):
        self.__r_w += g_w ** 2
        self.__r_b += g_b ** 2

        return self.learning_rate * g_w / (np.sqrt(self.__r_w) + 1e-7), self.learning_rate * g_b / (np.sqrt(self.__r_b) + 1e-7)

class rmsprop(optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)
        self.__alpha = 0.9
        self.__r_w = 0
        self.__r_b = 0

    def optimize(self, g_w, g_b):
        self.__r_w = self.__alpha * self.__r_w + (1 - self.__alpha) * g_w ** 2
        self.__r_b = self.__alpha * self.__r_b + (1 - self.__alpha) * g_b ** 2

        return self.learning_rate * g_w / (np.sqrt(self.__r_w) + 1e-7), self.learning_rate * g_b / (np.sqrt(self.__r_b) + 1e-7)

class adam(optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)
        self.__t = 0
        self.__alpha = 0.9
        self.__alpha2 = 0.999
        self.__s_w = 0
        self.__s_b = 0
        self.__r_w = 0
        self.__r_b = 0

    def optimize(self, g_w, g_b):
        self.__t += 1

        self.__s_w = self.__alpha * self.__s_w + (1 - self.__alpha) * g_w
        s_hat_w = self.__s_w / (1 - self.__alpha ** self.__t)
        self.__r_w = self.__alpha2 * self.__r_w + (1 - self.__alpha2) * np.power(g_w, 2)
        r_hat_w = self.__r_w / (1 - self.__alpha2 ** self.__t)
        v_w = s_hat_w / (1e-8 + np.sqrt(r_hat_w))

        self.__s_b = self.__alpha * self.__s_b + (1 - self.__alpha) * g_b
        s_hat_b = self.__s_b / (1 - self.__alpha ** self.__t)
        self.__r_b = self.__alpha2 * self.__r_b + (1 - self.__alpha2) * np.power(g_b, 2)
        r_hat_b = self.__r_b / (1 - self.__alpha2 ** self.__t)
        v_b = s_hat_b / (1e-8 + np.sqrt(r_hat_b))

        return self.learning_rate * v_w, self.learning_rate * v_b