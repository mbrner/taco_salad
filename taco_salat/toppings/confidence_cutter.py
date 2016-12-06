#!/usr/bin/env python
# -*- coding: utf-8 -*-
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from sklearn.metrics import confusion_matrix

from scipy.optimize import minimize

from IPython import embed

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
            idx = np.logical_and(self.edges[:, 0] <= x_i,
                                 self.edges[:, 1] > x_i)
            y_values[i] = np.mean(self.y[idx])
        return Curve(switch_points, y_values, mode=mode)

    def evaluate(self, x):
        return self.curve(x)

    def __call__(self, x):
        return self.evaluate(x)


def purity_criteria(threshold=0.99):
    """Returns decisison function which returns if the desired purity
    is fulfilled.
    Parameters
    ----------
    threshold : float or callable
        If float, independent from the positon of the window a constant
        criteria, which has to be fulfilled for each windowd, is used.
        If callable the function has to take the position and return a
        criteria not greater than 1.
    Returns
    -------
    decisison function : callable
        Returns a func(y_true, y_pred, position, sample_weights=None)
        returning 'fulfilled' which indicated if the desired purity is
        fulfilled.
    """
    if isinstance(threshold, float):
        if threshold > 1.:
            raise ValueError('Constant threshold must be <= 1')
        threshold_func = lambda x: threshold
    elif callable(threshold):
        threshold_func = threshold
    else:
        raise TypeError('\'threshold\' must be either float or callable')

    def decision_function(y_true, y_pred, position, sample_weights=None):
        """Returns decisison function which returns if the desired purity
        is fulfilled.
        Parameters
        ----------
        y_true : 1d array-like
            Ground truth (correct) target values. Only binary
            classification is supported.

        y_pred : 1d array-like
            Estimated targets.

        position : float
            Value indicating the postion of the cut window.

        Returns
        -------
        fulfilled : boolean
            Return if the criteria is fulfilled.
        """
        float_criteria = threshold_func(position)
        if not isinstance(float_criteria, float):
            raise TypeError('Callable threshold must return float <= 1.')
        if float_criteria > 1.:
            raise ValueError('Callable threshold returned value > 1')
        y_true_bool = np.array(y_true, dtype=bool)
        y_pred_bool = np.array(y_pred, dtype=bool)
        if sample_weights is None:
            tp = np.sum(y_true_bool[y_pred_bool])
            fp = np.sum(~y_true_bool[y_pred_bool])
        else:
            idx_tp = np.logical_and(y_true_bool, y_pred_bool)
            idx_fp = np.logical_and(~y_true_bool, y_pred_bool)
            tp = np.sum(sample_weights[idx_tp])
            fp = np.sum(sample_weights[idx_fp])
        if tp + fp == 0:
            purity = 0.
        else:
            purity = tp / (tp + fp)
        return np.absolute(purity - float_criteria)


    return decision_function


class ConfidenceCutter(object):
    """Method to find a confidence (X_c) cut curve depended on a second
    observable (X_o).

    Parameters
    ----------
    window_size: float
        Size of the sliding window

    n_steps : integer, optional (default=1000)
        Number of steps for the sliding window

    n_bootstraps : integer (default=3)
        Number of bootstraps. The curve is determined on n_boostraps
        samples and averaged.

    criteria : callable, optional (default=purity_criteria(threshold=0.99))
        Funtion criteria(y_true, y_pred, position, sample_weights=None)
        return True/False if the criteria is fulfilled.
        See e.g. purity_criteria

    positions : 1d array-lke, optional (default=None)
        If no positions are provided n_steps equally distributed between
        X_o min/max are used. The positions are mainly used for energy
        dependent decision functions

    conf_index : integer, optional (default=0)
        Index of the confidence in the X [n_samples, 2] used in all
        the functions. Default: X_c = X[:, 0] & X_o[:, 0]

    Attributes
    ----------
    cut_opts : Object CutOpts
        Container object containing all the options regarding the
        liding window

    criteria : callable
        See Parameters criteria

    conf_index : int or list
        See Parameters conf_index"""
    def __init__(self,
                 n_steps=1000,
                 window_size=0.1,
                 n_bootstraps=3,
                 criteria=purity_criteria(threshold=0.99),
                 positions=None,
                 conf_index=0,
                 n_jobs=0):
        self.cut_opts = self.CutOpts(n_steps=n_steps,
                                     window_size=window_size,
                                     n_bootstraps=n_bootstraps,
                                     positions=positions)
        self.criteria = criteria
        self.conf_index = conf_index
        self.n_jobs = n_jobs

    class CutOpts(object):
        """Class dealing with all settings of the sliding window.

        Parameters
        ----------
        window_size: float
            Size of the sliding window

        n_steps : integer, optional (default=1000)
            Number of steps for the sliding window

        n_bootstraps : integer (default=3)
            Number of bootstraps. The curve is determined on n_boostraps
            samples and averaged.. The curve is determined on n_boostraps
                samples and averaged.

        criteria : callable, optional (default=purity_criteria(threshold=0.99))
            Funtion criteria(y_true, y_pred, position, sample_weights=None)
            return True/False if the criteria is fulfilled.
            See e.g. purity_criteria

        positions : 1d array-lke, optional (default=None)
            If no positions are provided n_steps equally distributed between
            X_o min/max are used.

        position_type : ['mid', 'mean'], optional (default='mid')
            Type of the postions to the windows. If positions are provided,
            they are used to 'seed' the windows and for further calculations,
            the postions are reevaluated according to the selected type.

        Attributes
        ----------
        window_size: float
            See Parameters window_size

        n_steps : integer
            See Parameters n_steps

        n_bootstraps : integer
            See Parameters n_bootstraps

        criteria : callable
            See Parameters criteria

        positions : 1d array-lke
            See Parameters positions

        position_type : ['mid', 'mean']
            See Parameters position_type
            """
        def __init__(self,
                     n_steps=1000,
                     window_size=0.1,
                     n_bootstraps=10,
                     positions=None,
                     curve_type='mid'):
            self.n_steps = n_steps
            self.n_bootstraps = n_bootstraps
            self.window_size = window_size
            self.edges = None
            self.positions = positions
            self.curve_type = curve_type
            self.cut_curve = None

        def init_sliding_windows(self, X_o=None, sample_weight=None):
            """Initilizes the sliding windows.

            Uses the X_o values to determine the edges and the postions
            of all the different windows. If positions were provided in
            the __init__ method, those are used to determine the edges.
            If not the windows are equally distributed between min(X_o)
            and max(X_o). When the edges are set, the positions are
            evaluated according to the position_type set in the
            constructor.

            Parameters
            ----------
            X_o : array-like, shape=(n_samples)
                The input samples. 1d array with the X_o values.

            sample_weight : array-like, shape=(n_samples)
                The input samples. 1d array with the weights.

            Returns
            -------
            edges : array of shape=(n_steps, 2)
                Returns the edges of the sliding windows.
                Lower edges = edges[:, 0]
                Upper edges = edges[:, 1]

            positions : array of shape=(n_steps)
                Returns positions corresponding to the sliding windows.
                See Paremeters positions_type for more infos.
            """
            h_width = self.window_size / 2.
            if self.positions is None:
                assert X_o is not None, 'If not positions are provided, the ' \
                                        'the sliding windows has to be '\
                                        'initiated with data (X_0 != None)'

                min_e = np.min(X_o)
                max_e = np.max(X_o)
                self.positions = np.linspace(min_e + h_width,
                                             max_e - h_width,
                                             self.n_steps)
            self.edges = np.zeros((self.n_steps, 2))
            self.edges[:, 0] = self.positions - h_width
            self.edges[:, 1] = self.positions + h_width

        def generate_cut_curve(self, cut_values):
            """Evaluating all cut values and edges.

            And generates the points of which the cut curve consists.
            Mainly the overlapping of the different windows and the
            averaging of multiple bootsstraps is handeled

            Parameters
            ----------
            cut_values : array-like, shape=(n_steps, n_bootstraps)
                cut_values for each window and bootstraps if
                n_bootstraps > 1

            Returns
            -------
            curve_values : 1-d array
                Values defining the final cut curve
            """
            if self.n_bootstraps > 0:
                cut_values = np.mean(cut_values, axis=1)
            self.cut_curve = CurveSliding(self.edges, cut_values)
            return self.cut_curve

    def predict(self, X):
        """Predict class for X.
        The predicted class of an input sample is a vote by the trees in
        the forest, weighted by their probability estimates. That is,
        the predicted class is the one with highest mean probability
        estimate across the trees.
        Parameters
        ----------
        X : array-like shape=(n_samples, 2)
            The input samples. With the confidence at index 'conf_index'
            (default=0) and X_o as the other index.
        Returns
        -------
        y_pred : array of shape = [n_samples]
            The predicted classes.
        """
        if self.conf_index == 1:
            new_X = np.zeros_like(X)
            new_X[:, 1] = X[:, 0]
            new_X[:, 0] = X[:, 1]
            X = new_X
        X_o = X[:, 1]
        X_c = X[:, 0]
        thresholds = self.cut_opts.cut_curve(X_o)
        y_pred = np.array(X_c >= thresholds, dtype=int)
        return y_pred

    def fit(self, X, y, sample_weight=None):
        """Fit estimator.
        Parameters
        ----------
        X : array-like shape=(n_samples, 2)
            The input samples. With the confidence at index 'conf_index'
            (default=0) and X_o as the other index.

        y : array-like, shape=(n_samples)
            The target values (class labels in classification,
            possible_classed=[0, 1]).

        sample_weight : array-like, shape=(n_samples) or None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Returns self.
        """
        assert X.shape[1] == 2, 'X must have the shape (n_events, 2)'
        assert len(y) == X.shape[0], 'len(X) and len(y) must be the same'
        if sample_weight is not None:
            assert len(y) == len(sample_weight), 'weights and y need the' \
                'same length'

        if self.conf_index == 1:
            new_X = np.zeros_like(X)
            new_X[:, 1] = X[:, 0]
            new_X[:, 0] = X[:, 1]
            X = new_X

        n_events = X.shape[0]
        self.cut_opts.init_sliding_windows(X[:, 1], sample_weight)
        n_bootstraps = self.cut_opts.n_bootstraps
        if n_bootstraps is None or n_bootstraps <= 0:
            cut_values = self.__determine_cut_values__(
                X, y, sample_weight)
        else:
            cut_values = np.zeros((self.cut_opts.n_steps, n_bootstraps))
            for i in range(self.cut_opts.n_bootstraps):
                bootstrap = np.random.choice(n_events, n_events, replace=True)
                bootstrap = np.sort(bootstrap)
                X_i = X[bootstrap]
                y_i = y[bootstrap]
                if sample_weight is None:
                    sample_weight_i = None
                else:
                    sample_weight_i = sample_weight[bootstrap]
                cut_values[:, i] = self.__determine_cut_values_mp__(
                    X_i, y_i, sample_weight_i)
        self.cut_opts.generate_cut_curve(cut_values)
        return self

    def __determine_cut_values__(self, X, y_true, sample_weight):
        edges = self.cut_opts.edges
        positions = self.cut_opts.positions
        cut_values = np.zeros_like(positions)
        X_o = X[:, 1]
        X_c = X[:, 0]
        for i, [[lower, upper], position] in enumerate(zip(edges, positions)):
            idx = np.logical_and(X_o >= lower, X_o < upper)
            confidence_i = X_c[idx]
            y_true_i = y_true[idx]
            weights_i = None
            if sample_weight is not None:
                weights_i = sample_weight[idx]

            def wrapped_decision_func(cut):
                y_pred_i_j = np.array(confidence_i >= cut, dtype=int)
                return self.criteria(y_true_i,
                                     y_pred_i_j,
                                     position,
                                     sample_weights=weights_i)

            possible_cuts = np.unique(confidence_i)
            min_idx = min(possible_cuts, key=wrapped_decision_func)
            cut_values[i] = min_idx
        return cut_values


    def __find_best_cut__(self, i, lower, upper, position,
                          X, y_true, sample_weight, n_points=10):
        X_o = X[:, 1]
        X_c = X[:, 0]
        idx = np.logical_and(X_o >= lower, X_o < upper)
        confidence_i = X_c[idx]
        y_true_i = y_true[idx]
        weights_i = None
        if sample_weight is not None:
            weights_i = sample_weight[idx]

        possible_cuts = np.sort(np.unique(confidence_i))

        def wrapped_decision_func(cut):
            y_pred_i_j = np.array(confidence_i >= cut, dtype=int)
            return self.criteria(y_true_i,
                                 y_pred_i_j,
                                 position,
                                 sample_weights=weights_i)

        cut_value = self.__find_best_cut_2__(wrapped_decision_func,
                                             possible_cuts,
                                             n_points=n_points)
        return cut_value

    def __find_best_cut_list__(self, arg):
        i=arg[0]
        lower=arg[1]
        upper=arg[2]
        position=arg[3]
        X=arg[4]
        y_true=arg[5]
        sample_weight=arg[6]
        n_points=arg[7]
        return self.__find_best_cut__(i, lower, upper, position,
                                      X, y_true, sample_weight, n_points)

    def __find_best_cut_2__(self, eval_func, conf, n_points=100):
        n_confs = len(conf)
        step_width = int(n_confs / (n_points - 1))
        if step_width == 0:
            final = True
            selected_cuts = conf
        else:
            final = False
            idx = [(i * step_width) for i in range(n_points)]
            idx[-1 ] = n_confs -1
            selected_cuts = conf[idx]
        criteria_values = [eval_func(cut) for cut in selected_cuts]
        idx_min = np.argmin(criteria_values)
        if final:
            return selected_cuts[idx_min]
        else:
            if idx_min == 0:
                conf = conf[:idx[idx_min + 1]]
            elif idx_min == n_points - 1:
                conf = conf[idx[idx_min - 1]:]
            else:
                conf = conf[idx[idx_min - 1]:idx[idx_min + 1]]
            return self.__find_best_cut_2__(eval_func, conf, n_points)

    def __determine_cut_values_mp__(self, X, y_true, sample_weight):
        edges = self.cut_opts.edges
        positions = self.cut_opts.positions

        X_o = X[:, 1]
        X_c = X[:, 0]
        n_points = 5
        if self.n_jobs > 1:
            tasks = []
            for i, [[lower, upper], position] in enumerate(zip(edges,
                                                               positions)):
                tasks.append([i, lower, upper, position,
                              X, y_true, sample_weight, n_points])
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                chunksize = int(len(tasks) / self.n_jobs)
                cut_values = executor.map(self.__find_best_cut_list__,
                                          tasks,
                                          chunksize=chunksize)
                cut_values = np.array([cut_i for cut_i in cut_values])
        else:
            cut_values = np.zeros_like(positions)
            for i, [[lower, upper], position] in enumerate(zip(edges,
                                                               positions)):
                cut_value = self.__find_best_cut__(i, lower, upper, position,
                                                   X, y_true, sample_weight,
                                                   n_points=n_points)

                cut_values[i] = cut_value
        return cut_values



    def get_params():
        pass


if __name__ == '__main__':
    from test_conf_cutter import generate
    from matplotlib import pyplot as plt
    purity = 0.9
    x, y_pred, y_true = generate(10000, purity)
    X = np.vstack((x, y_pred))

    conf_cutter = ConfidenceCutter(n_steps=100,
                                   window_size=0.3,
                                   n_bootstraps=10,
                                   criteria=purity_criteria(threshold=0.95),
                                   conf_index=1,
                                   n_jobs=0)
    conf_cutter.fit(X.T, y_true)
    X = np.linspace(-1., 1., 1000)
    Y = conf_cutter.cut_opts.cut_curve(X)
    plt.hexbin(y_pred, x, gridsize=30)
    plt.plot(Y, X, color='k', lw='5')
    plt.show()




