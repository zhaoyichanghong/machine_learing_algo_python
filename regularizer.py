import numpy as np

class regularizer:
    def __init__(self, param):
        self.param = param

    def regularize(self, W):
        return 0

class l1_regularizer(regularizer):
    def __init__(self, param):
        super().__init__(param)

    def regularize(self, W):
        return self.param * np.where(W >= 0, 1, -1)

class l2_regularizer(regularizer):
    def __init__(self, param):
        super().__init__(param)

    def regularize(self, W):
        return self.param * W

class elastic_regularizer(regularizer):
    def __init__(self, param, ratio):
        self.__l1 = l1_regularizer(param)
        self.__l2 = l2_regularizer(param)
        self.__ratio = ratio

    def regularize(self, W):
        return self.__ratio * self.__l1.regularize(W) + (1 - self.__ratio) * self.__l2.regularize(W)