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