import numpy as np

class Regularizer:
    def __init__(self, param):
        self._param = param

    def regularize(self, W):
        return 0

class L1Regularizer(Regularizer):
    def __init__(self, param):
        super().__init__(param)

    def regularize(self, W):
        return self._param * np.where(W >= 0, 1, -1)

class L2Regularizer(Regularizer):
    def __init__(self, param):
        super().__init__(param)

    def regularize(self, W):
        return self._param * W

class ElasticRegularizer(Regularizer):
    def __init__(self, param, ratio):
        self.__l1 = L1Regularizer(param)
        self.__l2 = L2Regularizer(param)
        self.__ratio = ratio

    def regularize(self, W):
        return self.__ratio * self.__l1.regularize(W) + (1 - self.__ratio) * self.__l2.regularize(W)