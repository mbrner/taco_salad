#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .recipe import Recipe

class TacoSalat(Recipe):
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
        super(TacoSalat, self).__init__()
        if df is not None and roles is not None:
            columns = list(df.columns)
            for name, role in zip(columns, roles):
                if role in [0, 1, 2, 3]:
                    self.add_ingredient(unique_name=name,
                                               name_layer=name,
                                               role=role)
        else:
            for att in attributes:
                self.add_ingredient(unique_name=att,
                                           name_layer=att,
                                           role=0)
            for label in labels:
                self.add_ingredient(unique_name=att,
                                           name_layer=att,
                                           role=1)
            for misc_i in misc:
                self.add_ingredient(unique_name=att,
                                           name_layer=att,
                                           role=3)
            for weight_i in weights:
                self.add_ingredient(unique_name=weight_i,
                                           name_layer=weight_i,
                                           role=2)

    def add_component(self,
                      layer
                      name=None,
                      clf=None,
                      attributes=None,
                      label=None,
                      returns=None,
                      weight=None,
                      comment=''):
        long_name_attributes = []
        for att in attributes:
            selected_att = self.get(att)






        component = super(TacoSalat, self).add_component(layer=layer,
                                                         name=name,
                                                         comment=comment)






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

    def add_component(self,
                      name,
                      clf=None,
                      attributes=None,
                      label=None,
                      returns=None,
                      weight=None,
                      comment=')

