# -*- coding:utf-8 -*-
"""
Methods to treat a curve defined by a finite number ob (x, y) pairs
as a continious y=f(x).
"""
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
        self.curve = self.setup_curve(mode)

    def setup_curve(self, mode):
        switch_points = np.sort(np.unique(self.edges))[:-1]
        y_values = np.zeros_like(switch_points)
        for i, x_i in enumerate(switch_points):
            idx = np.logical_and(edges[:, 0] <= x_i,
                                 edges[:, 1] > x_i)
            y_values[i] = np.mean(self.y[idx])
        return Curve(switch_points, y_values, mode=mode)

    def evaluate(self, x):
        return self.curve(x)

    def __call__(self, x):
        return self.evaluate(x)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    n = np.random.normal(size=10000)
    binning = np.linspace(-3., 3., 21)
    hist = np.histogram(n, bins=binning)[0]
    x = (binning[1:] + binning[:-1]) / 2.





    c_1 = Curve(x, hist, mode='hist')
    c_2 = Curve(x, hist, mode='linear')


    n_steps = 20
    h_width = 0.4
    positions = np.linspace(-4. + h_width,
                                    4. - h_width,
                                    n_steps)
    edges = np.zeros((n_steps, 2))
    edges[:, 0] = positions - h_width
    edges[:, 1] = positions + h_width

    y_values = np.zeros(n_steps)
    for i, [[lower, upper], position] in enumerate(zip(edges, positions)):
        idx = np.logical_and(n >= lower, n < upper)
        y_values[i] = np.sum(idx)

    print(y_values)

    c_3 = CurveSliding(edges, y_values, mode='hist')
    c_4 = CurveSliding(edges, y_values, mode='linear')


    x = np.linspace(-5, 5, 1000)
    y_1 = c_1(x)
    y_2 = c_2(x)
    y_3 = c_3(x)
    y_4 = c_4(x)




    plt.plot(x, y_1)
    plt.plot(x, y_2)
    plt.plot(x, y_3)
    plt.plot(x, y_4)

    plt.show()









