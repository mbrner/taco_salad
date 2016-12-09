#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

class BaseComponent(object):
    def __init__(self,
                 name,
                 comment=''):
        self.name = name
        self.comment = comment


class Component(BaseComponent):
    def __init__(self,
                 name,
                 clf=None,
                 attributes=None,
                 label=None,
                 returns=None,
                 weight=None,
                 fit_func='fit',
                 predict_func='predict_proba',
                 comment=''):
        super(Component, self).__init__(name=name,
                                        comment=comment)
        self.clf = clf
        self.attributes = attributes
        self.label = label
        self.returns = returns
        self.weight = weight
        self.predict_func = getattr(clf, predict_func)
        self.fit_func = getattr(clf, fit_func)

    def fit_df(self, df):
        X = df.loc[:, self.attributes]
        y = df.loc[:, self.label]
        if self.weight is None:
            self.clf = self.fit_func(X.values, y.values)
        else:
            sample_weight = df.loc[:, self.weight]
            self.clf = self.clf.fit(X.values, y.values, sample_weight)
        return self

    def predict_df(self, df):
        idx = df.index
        X = df.loc[:, self.attributes]
        return_clf = self.predict_func(X.values)
        return_df = pd.DataFrame(columns=self.returns,
                                 index=idx)
        if len(self.returns) == 1:
            return_df[self.returns[0]] = return_clf
        else:
            for i, name in enumerate(self.returns):
                return_df[name] = return_clf[:, i]
        return return_df


