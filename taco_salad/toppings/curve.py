#!/usr/bin/env python
# -*- coding: utf-8 -*-
from copy import deepcopy
import numpy as np


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
        self.mode = mode.lower()
        assert self.mode in ['hist', 'linear'], \
            'Invalid mode [\'hist\', \'linear\']'
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

    def __add__(self, other):
        copy = deepcopy(self)
        if isinstance(other, Curve):
            assert len(self.x) == len(other.x), 'Only curves with same x ' \
                                                'can be added.'
            assert (self.x == other.x).all(), 'Only curves with same x ' \
                                              'can be added.'
            assert other.mode == self.mode, 'Only curves with same mode' \
                                            'can be added'

            if self.mode == 'hist':
                copy.y += other.y
            elif self.mode == 'linear':
                copy.slope += other.slope
                copy.offset += other.offset
        else:
            copy = deepcopy(self)
            value = float(other)
            if self.mode == 'hist':
                copy.y += value
            elif self.mode == 'linear':
                copy.offset += value
        return copy

    def __sub__(self, other):
        if isinstance(other, Curve):
            assert len(self.x) == len(other.x), 'Only curves with same x ' \
                                                'can be subtracted.'
            assert (self.x == other.x).all(), 'Only curves with same x ' \
                                              'can be subtracted.'
            assert other.mode == self.mode, 'Only curves with same mode' \
                                            'can be subtracted'
        return self.__add__(other * (-1))

    def __mul__(self, value):
        value = float(value)
        copy = deepcopy(self)
        if self.mode == 'hist':
            copy.y *= value
        elif self.mode == 'linear':
            copy.slope *= value
            copy.offset *= value
        return copy

    def __div__(self, value):
        value = float(value)
        return self.__mul__(1. / value)


class CurveSliding(Curve):
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
    def __init__(self,
                 edges,
                 y_input,
                 mode='linear',
                 combination_mode='overlapping'):
        self.combination_mode = combination_mode
        self.mode = mode.lower()
        assert self.mode in ['hist', 'linear'], \
            'Invalid mode [\'hist\', \'linear\']'
        assert self.combination_mode in ['overlapping', 'single'], \
            'Invalid mode [\'overlapping\', \'single\']'
        self.x, self.y = self.setup_x_y(edges, y_input)
        self.setup_curve(mode)

    def setup_x_y(self, edges, y_input):
        if self.combination_mode == 'overlapping':
            switch_points = np.sort(np.unique(edges))[:-1]
            y_values = np.zeros_like(switch_points)
            for i, x_i in enumerate(switch_points):
                idx = np.logical_and(edges[:, 0] <= x_i,
                                     edges[:, 1] > x_i)
                y_values[i] = np.mean(y_input[idx])
            return switch_points, y_values
        elif self.combination_mode == 'single':
            window_mids = edges[:, 1] - edges[:, 0]
            return window_mids, y_values
