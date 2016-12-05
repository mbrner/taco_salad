# -*- coding:utf-8 -*-
"""
Methods to treat a curve defined by a finite number ob (x, y) pairs
as a continious y=f(x).
"""
import numpy as np

class curve:
    """Class to treat a curve defined by a finite number ob (x, y) pairs
    as a continious y=f(x).

    Parameters
    ----------
    x : np.array, shape=(N)
        X values.

    y : np.array, shape=(N)
        Y values.

    mode : ['linear', 'hist'], optional (default='linear')
        Mode in which the (x,y) are used to generate the curve:

        hist = stepwise, only input y are possible returns when evaluted
        linear = linear interpolation between neigbouring X values.
    """
    def __init__(self, x, y, mode='linear'):
        order = np.argsort(x)
        self.x = x[order]
        self.y = y[order]
        self.setup_curve(mode)

    def setup_curve(self, mode):
        if mode == 'hist':
            self.evaluate = self.__eval_hist__
            self.__setup_hist__()
        if mode == 'linear':
            self.evaluate = self.__eval_linear__
            self.__setup_linear__()

    def __call__(self, x):
        return self.evaluate(x)

    def __eval_linear__(self, x):
        digitized = np.digitize(x, self.x)
        return x * self.slope[digitized] + self.offset[digitized]

    def __setup_linear__(self):
        slope = (self.y[1:] - self.y[:-1]) / (self.x[1:] - self.x[:-1])
        self.slope = np.zeros(len(self.x) + 1)
        self.slope[1:-1] = slope
        self.offset = np.hstack((y[0], y))
        offset = y[:-1] - slope * x[:-1]
        self.offset[1:-1] = offset

    def __eval_hist__(self, x):
        digitized = np.digitize(x, self.edges)
        return self.y[digitized]

    def __setup_hist__(self):
        self.edges = (self.x[1:] + self.x[:-1]) / 2.
