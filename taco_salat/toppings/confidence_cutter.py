#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd


class ConfidenceCutter(object):
    class CutOpts(object):
        def __init__(self,
                     n_steps=1000,
                     window_size=0.1,
                     n_bootstraps=10):
            self.n_steps = n_steps
            self.window_size = window_size
            self.n_bootstraps = n_bootstraps

    def __init__(self,
                 n_steps=1000,
                 window_size=0.1,
                 n_bootstraps=3,
                 criteria='purity',
                 threshold=0.99,
                 positions=None,
                 mode='steps'):
        self.cut_opts = self.CutOpts(n_steps=n_steps,
                                     window_size=window_size,
                                     n_bootstraps=n_bootstraps)
        if criteria == 'purity':
            self.__fit__ = self.__fit_purity__
        self.threshold = threshold
        self.positions = positions
        self.cut_curve = None

    def fit(self, X, y, weights=None, conf_index=0):
        if weights is None:
            weights = np.ones_like(y)
        assert X.shape[1] == 2, 'X must have the shape (n_events, 2)'
        assert len(y) == X.shape[0], 'len(X) and len(y) must be the same'
        assert len(y) == len(weights), 'weights and y need the same length'
        width = self.cut_opts.window_size
        if conf_index == 1:
            new_X = np.zeros_like(X)
            new_X[:, 1] = X[:, 0]
            new_X[:, 0] = X[:, 1]
            X = new_X
        n_events = X.shape[0]
        y[y == 2] = 0
        if self.positions is None:
            min_e = np.min(X[:, 1])
            max_e = np.max(X[:, 1])
            self.positions = np.linspace(min_e + (width / 2.),
                                         max_e - (width / 2.),
                                         self.cut_opts.n_steps)
        n_bootstraps = self.cut_opts.n_bootstraps
        cut_curve_df = pd.DataFrame(index=self.positions,
                                    columns=range(n_bootstraps))
        for i in range(n_bootstraps):
            train = np.random.choice(n_events, n_events, replace=True)
            train = np.sort(train)
            X_i = X[train]
            y_i = y[train]
            w_i = weights[train]
            cut_values = self.__fit__(X_i, y_i, w_i)
            cut_curve_df.loc[self.positions, i] = cut_values
        cut_curve_df['average'] = 0.
        for i, p_i in enumerate(self.positions):
            selection = np.absolute(self.positions - p_i) <= width / 2.
            idx = self.positions[selection]
            mean = np.mean(cut_curve_df.loc[idx, range(n_bootstraps)].values)
            cut_curve_df.loc[p_i, 'average'] = mean
        width = self.cut_opts.window_size / 2.
        all_postions = np.unique(
            np.hstack((self.positions - width,
                      self.positions + width,
                      self.positions)))
        combined_curve_df = pd.DataFrame(index=all_postions,
                                         columns=['cut_value'])
        for p_i in all_postions:
            mask_above = cut_curve_df.loc[:, 'average'] - p_i <= width / 2.
            mask_below = p_i - cut_curve_df.loc[:, 'average'] < width / 2.
            mask = np.logical_or(mask_above, mask_below)
            cut_val = np.mean(cut_curve_df.loc[mask, 'average'])
            combined_curve_df.loc[p_i, 'cut_value'] = cut_val
        self.cut_curve = combined_curve_df

    def __fit_purity__(self, X, y, weights):
        width = self.cut_opts.window_size
        cut_series = pd.Series(0., index=self.positions)
        is_signal = y == 1
        is_backgorund = y == 0
        for i, p_i in enumerate(self.positions):
            in_window = np.absolute(X[:, 1] - p_i) <= width / 2.
            signal_in_window = np.logical_and(in_window, is_signal)
            background_in_window = np.logical_and(in_window, is_backgorund)

            sig_conf = X[signal_in_window, 0]
            bkg_conf = X[background_in_window, 0]
            sig_w = weights[signal_in_window]
            bkg_w = weights[background_in_window]

            possible_cuts = np.unique(X[in_window, 0])
            possible_cuts = np.sort(possible_cuts)[::-1]
            last_cut = 1.
            for cut in possible_cuts:
                sum_w_sig = np.sum(sig_w[sig_conf >= cut])
                sum_w_bkg = np.sum(bkg_w[bkg_conf >= cut])
                purity = sum_w_sig / (sum_w_sig + sum_w_bkg)
                if purity < self.threshold:
                    break
                else:
                    last_cut = cut
            cut_series[p_i] = last_cut
        return cut_series

    def predict(self, conf, en, mode='steps'):
        prediction = np.zeros_like(en, dtype=int)
        cut_index = np.digitize(en, bins=self.cut_curve.index)
        for i, [c_i, v_i] in enumerate(zip(conf, cut_index)):
            if c_i >= v_i:
                prediction[i] = 1
            else:
                prediction[i] = 0
        return prediction

