#!/usr/bin/env python
# -*- coding: utf-8 -*-
from innards import Recipe

class TacoSalat(object):
    """Base Class providing an interface to stack classification layers.

    Parameters
    ----------
    ingredients : list of column names or int
        Instance of LabelFactory used to generate Label from the Data.

    Attributes
    ----------

    """
    def __init__(self,
                 df=None,
                 roles=None,
                 attributes=[],
                 labels=[],
                 weights=[],
                 misc=[]):
        self.recipe = Recipe()
        if df is not None and roles is not None:
            columns = list(df.columns)
            for name, role in zip(columns, roles):
                if role in [0, 1, 2, 3]:
                    self.recipe.add_ingredient(unique_name=name,
                                               name_layer=name,
                                               role=role)
        else:
            for att in attributes:
                self.recipe.add_ingredient(unique_name=att,
                                           name_layer=att,
                                           role=0)
            for label in labels:
                self.recipe.add_ingredient(unique_name=att,
                                           name_layer=att,
                                           role=1)
            for misc_i in misc_i:
                self.recipe.add_ingredient(unique_name=att,
                                           name_layer=att,
                                           role=3)
            for weight_i in weights:
                self.recipe.add_ingredient(unique_name=weight_i,
                                           name_layer=weight_i,
                                           role=2)


    def fit_df(df, type_col='Type', weight_col='Weight'):
        raise NotImplementedError

    def predict_proba_df(df):
        raise NotImplementedError








    def predict_df(df):
        raise NotImplementedError

    def fit(X, y, sample_weights=None):
        raise NotImplementedError

    def predict(X):
        raise NotImplementedError

    def predict_proba(X):
        raise NotImplementedError

