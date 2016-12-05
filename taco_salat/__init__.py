#!/usr/bin/env python
# -*- coding: utf-8 -*-


class TacoSalat(object):
    """Base Class providing an interface to stack classification layers.

    Parameters
    ----------
    label_factorty: instance of utensils.LabelFactory
        Instance of LabelFactory used to generate Label from the Data.

    Attributes
    ----------

    """
    def __init__(self, label_factory):
        self.label_factory = label_factory

    def fit(X, y, sample_weights=None):
        raise NotImplementedError

    def predict(X):
        raise NotImplementedError

    def predict_proba(X):
        raise NotImplementedError

    def fit_df(df, type_col='Type', weight_col='Weight'):
        raise NotImplementedError

    def predict_df(df):
        raise NotImplementedError

    def predict_proba_df(df):
        raise NotImplementedError

