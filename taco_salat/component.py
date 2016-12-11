#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd


class BaseComponent(object):
    """Basic Component.

    Parameters
    ----------
    name : str
        Name of the Component.

    comment : str, optional (default='')
        Commet for better documentation of the architecture.

    """
    def __init__(self,
                 name,
                 comment=''):
        self.name = name
        self.comment = comment


class Component(BaseComponent):
    """Component with classifier/regressor input.

    Parameters
    ----------
    name : str
        Name of the component.

    clf : object
        Object providing fit and predict methods. Syntax is compatible
        with the sklearn classifier (fit(X, y, sample_weight),
        predict(X).

    attributes : list of str
        List of names of the features used for training.

    label : str
        Name of the label feature.

    returns : List of str
        Names of the returned features.

    weight : str, optional (default=None)
        Name of the weight.

    fit_func : str, optional (default'fit')
        Name of the callable used in the fit_df function for the
        component.

    predict_func : str, optional (default'predict_proba')
        Name of the callable used in the predict_df function for the
        component.

    comment : str, optional (default='')
        Commet for better documentation of the architecture.

    Attributes
    ----------
    name : str
        Name of the component.

    attributes : list of str
        List of names of the features used for training.

    label : str
        Name of the label feature.

    returns : List of str
        Names of the returned features.

    weight : str or None
        Name of the weight.

    fit_func : callable
        Callable used in the fit_df function for the component.

    predict_func : callable
        Callable used in the predict_df function for the component.

    comment : str
        Comment.

    """

    def __init__(self,
                 name,
                 clf,
                 attributes,
                 label,
                 returns,
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
        """Method to fit the components.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing all the features.

        Returns
        -------
        self : component.Component
            Self.

        """
        X = df.loc[:, self.attributes]
        y = df.loc[:, self.label]
        if self.weight is None:
            self.clf = self.fit_func(X.values, y.values)
        else:
            sample_weight = df.loc[:, self.weight]
            self.clf = self.clf.fit(X.values, y.values, sample_weight)
        return self

    def predict_df(self, df):
        """Method to predict samples from a dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing all the features.

        Returns
        -------
        return_df : pandas.DataFrame
            Dataframe of the scores.

        """
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
