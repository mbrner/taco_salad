#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

import numexpr as ne


def purity_criteria(threshold=0.99):
    """Returns decisison function which returns the absolute difference
    between the achieved and desired purity.
    Parameters
    ----------
    threshold : float or callable
        If float, independent from the positon of the window a constant
        criteria, which has to be fulfilled for each window, is used.
        If callable the function has to take the position and return a
        criteria not greater than 1.
    Returns
    -------
    decisison function : callable
        Returns a func(y_true, y_pred, position, sample_weights=None)
        returning 'difference' which is the absolute difference between
        the achieved and desired purity
    """
    if isinstance(threshold, float):
        if threshold > 1.:
            raise ValueError('Constant threshold must be <= 1')

        def threshold_func(x):
            return threshold
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


def general_confusion_matrix_criteria(eval_str, threshold=0.99):
    """Returns decisison function which returns the absolute difference
    the defined criteria and the threshold.

    Parameters
    ----------
    eval_str: string
        String for the criteria. The criteria is evaluated using
        the numexpr module. Usable values are:
            tp: True Positives
            fp: False Positives
            tn: True Negatives
            fn: False Negatives
        E.g.: Purity as the criteria: 'tp / (tp + fp)'

    threshold : float or callable
        If float, independent from the positon of the window a constant
        criteria, which has to be fulfilled for each window, is used.
        If callable the function has to take the position and return a
        criteria not greater than 1.

    Returns
    -------
    decisison function : callable
        Returns a func(y_true, y_pred, position, sample_weights=None)
        returning 'difference' which is the absolute difference between
        the achieved and desired purity
    """
    if isinstance(threshold, float):
        if threshold > 1.:
            raise ValueError('Constant threshold must be <= 1')

        def threshold_func(x):
            return threshold

    elif callable(threshold):
        threshold_func = threshold
    else:
        raise TypeError('\'threshold\' must be either float or callable')

    if 'tp' in eval_str:
        calc_tp = True
    else:
        calc_tp = False
    if 'fp' in eval_str:
        calc_fp = True
    else:
        calc_fp = False
    if 'tn' in eval_str:
        calc_tn = True
    else:
        calc_tn = False
    if 'fn' in eval_str:
        calc_fn = True
    else:
        calc_fn = False

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
        if calc_tp:
            if sample_weights is None:
                tp = np.sum(y_true_bool[y_pred_bool])
            else:
                idx_tp = np.logical_and(y_true_bool, y_pred_bool)
                tp = np.sum(sample_weights[idx_tp]) # NOQA

        if calc_fp:
            if sample_weights is None:
                fp = np.sum(~y_true_bool[y_pred_bool])
            else:
                idx_fp = np.logical_and(~y_true_bool, y_pred_bool)
                fp = np.sum(sample_weights[idx_fp]) # NOQA

        if calc_tn:
            if sample_weights is None:
                tn = np.sum(~y_true_bool[~y_pred_bool])
            else:
                idx_tn = np.logical_and(~y_true_bool, ~y_pred_bool)
                tn = np.sum(sample_weights[idx_tn]) # NOQA
        if calc_fn:
            if sample_weights is None:
                fn = np.sum(y_true_bool[~y_pred_bool])
            else:
                idx_fn = np.logical_and(y_true_bool, ~y_pred_bool)
                fn = np.sum(sample_weights[idx_fn]) # NOQA
        criteria_value = ne.evaluate(eval_str)
        return np.absolute(criteria_value - float_criteria)

    return decision_function
