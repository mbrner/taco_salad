#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from sklearn.metrics import confusion_matrix

from ..utensils.curve import CurveSliding

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
        confusion_matrix = confusion_matrix(y_true,
                                            y_pred,
                                            sample_weight=sample_weights)
        tp = confusion_matrix[1, 1]
        fp = confusion_matrix[0, 1]
        purity = tp / (tp + fp)
        fulfilled = purity >= float_criteria
        return fulfilled
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
                 conf_index=0):
        self.cut_opts = self.CutOpts(n_steps=n_steps,
                                     window_size=window_size,
                                     n_bootstraps=n_bootstraps,
                                     positions=positions)
        self.criteria = criteria
        self.conf_index = conf_index

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
                assert X_o not None, 'If not positions are provided, the ' \
                                     'the sliding windows has to be '\
                                     'initiated with data (X_0 != None)'

                min_e = np.min(X_o)
                max_e = np.max(X_o)
                self.positions = np.linspace(min_e + h_width,
                                             max_e - h_width,
                                             self.cut_opts.n_steps)
            self.edges = np.zeros((n_steps, 2))
            self.edges[:, 0] = initial_positions - h_width
            self.edges[:, 1] = initial_positions + h_width

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
        self.cut_opts.init_sliding_windows(X[:, 1],
                                                              sample_weight)
        n_bootstraps = self.cut_opts.n_bootstraps
        if n_bootstraps is None or n_bootstraps <= 0:
            cut_values = self.__determine_cut_values__(
                X, y, sample_weight)
        else:
            cut_values = np.zeros((self.cut_opts.n_steps,
                                                 n_bootstraps))
            for i in range(n_boostraps):
                bootstrap = np.random.choice(n_events, n_events, replace=True)
                bootstrap = np.sort(bootstrap)
                X_i = X[bootstrap]
                y_i = y[bootstrap]
                if sample_weight is None:
                    sample_weight_i = None
                else:
                    sample_weight_i = sample_weight[bootstrap]
                cut_values[:, i] = self.__determine_cut_values__(
                    X,_i y_i, sample_weight_i)
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
            weight_i = None
            if sample_weight is not None:
                weights_i = sample_weight[idx]

            possible_cuts = np.sort(np.unique(confidence_i))[::-1]
            selected_cut = conf_cuts[0]
            for cut_j in conf_cuts:
                y_pred_i_j = confidence_i >= cut_j
                if self.criteria(y_true_i,
                                 y_pred_i_j,
                                 position,
                                 sample_weights=weights_i)
                    selected_cut = cut_j
                else:
                    break
            cut_values[i] = cut_j
        return cut_values

    def get_params():
        pass


if __name__ == '__test__':






