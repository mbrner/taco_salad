#!/usr/bin/env python
# -*- coding: utf-8 -*-
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from sklearn.metrics import confusion_matrix

from scipy.optimize import minimize


class Curve:
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

    def setup_curve(self, mode='linear'):
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
        self.offset = np.hstack((self.y[0], self.y))
        offset = self.y[:-1] - slope * self.x[:-1]
        self.offset[1:-1] = offset

    def __eval_hist__(self, x):
        digitized = np.digitize(x, self.edges)
        return self.y[digitized]

    def __setup_hist__(self):
        self.edges = (self.x[1:] + self.x[:-1]) / 2.


class CurveSliding:
    """Class to treat a curve defined by a finite number ob (x, y) pairs
    as a continious y=f(x).

    Parameters
    ----------
    edges : np.array, shape=(n_steps, 2)
        Edges of the sliding windows:
            lower edges := edges[:, 0]
            upper edges := edges[:, 1]

    y : np.array, shape=(n_steps)
        Y values corresponding to the sliding windows.

    mode : ['linear', 'hist'], optional (default='linear')
        Mode in which the (x,y) are used to generate the curve:

        hist = stepwise, only input y are possible returns when evaluted
        linear = linear interpolation between neigbouring X values.
    """
    def __init__(self, edges, y, mode='linear'):
        self.edges = edges
        self.y = y
        self.mode = mode
        self.curve = self.setup_curve(mode)

    def setup_curve(self, mode='linear'):
        switch_points = np.sort(np.unique(self.edges))[:-1]
        y_values = np.zeros_like(switch_points)
        for i, x_i in enumerate(switch_points):
            idx = np.logical_and(self.edges[:, 0] <= x_i,
                                 self.edges[:, 1] > x_i)
            y_values[i] = np.mean(self.y[idx])
        return Curve(switch_points, y_values, mode=mode)

    def evaluate(self, x):
        return self.curve(x)

    def __call__(self, x):
        return self.evaluate(x)
