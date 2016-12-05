# -*- coding:utf-8 -*-
"""
Methods to treat a curve defined by a finite number ob (x, y) pairs
as a continious y=f(x).
"""
import numpy as np

class CutOpts(object):
    def __init__(self,
                 n_steps=1000,
                 window_size=0.1,
                 positions=None,
                 position_type='mid'):
        self.n_steps = n_steps
        self.window_size = window_size
        self.positions = positions
        self.positions_type = positions_type
        self.edges = None

    def init_sliding_windows(self, X_o=None, sample_weight=None):
        if self.positions is None:
            h_width = self.window_size / 2.
            min_e = np.min(X_o)
            max_e = np.max(X_o)
            initial_positions = np.linspace(min_e + h_width,
                                            max_e - h_width,
                                            self.n_steps)
        else:
            initial_positions = self.positions
            self.n_steps = len(self.positions)
        self.edges = np.zeros((self.n_steps, 2))
        self.edges[:, 0] = initial_positions - h_width
        self.edges[:, 1] = initial_positions + h_width
        if self.positions_type == 'mid':
            self.positions = initial_positions
        elif self.positions_type == 'mean':
            self.positions = np.zeros_like(initial_positions)
            for i, [upper, lower] in enumerate(self.edges):
                idx = np.logical(X_o >= lower, X_o < upper)
                sel_x = X_o[idx]
                if sample_weight is not None:
                    sel_w = sample_weight[idx]
                else:
                    sel_w = None
                self.positions[i] = np.average(sel_x, weights=sel_w)

    def generate_cut_curve(self, cut_values):
        if self.n_bootstraps > 0:
            cut_values = np.mean(cut_values, axis=1)
        self.cut_curve = CurveSliding(self.edges, cut_values)
        return self.cut_curve










