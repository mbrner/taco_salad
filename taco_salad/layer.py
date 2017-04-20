#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
from fnmatch import fnmatch
import logging

import pandas as pd
import numpy as np

from .component import BaseComponent


class BaseLayer(object):
    """Layer providing all functionality to buidl up the architecture.
    No fit or predict functions. Layer have a component_dict containing
    all the component objects.

    Parameters
    ----------
    name : str
        Name of the Layer.

    comment : str, optional (default='')
        Commet for better documentation of the architecture.

    Attributes
    ----------
    name : str
        Name of the layer.

    comment : str
        Comment.

    component_dict : dict
        Dictionary containing all components.

    n_components : int
        Number of components.

    """
    def __init__(self, name, comment=''):
        self.name = name
        self.comment = comment
        self.component_dict = {}
        self.n_components = 0
        self.active = False

    def add_component(self, component):
        assert component.name not in self.component_dict.keys(), \
            '{} already in layer {}'.format(component.name, self.name)
        self.n_components += 1
        self.component_dict[component.name] = component
        return self.component_dict[component.name]

    def get_component(self, name):
        return self.component_dict[name]

    def __getitem__(self, name):
        return self.get_component(name)

    def rename_component(self, component, new_name):
        if not isinstance(component, BaseComponent):
            component = self.get_component(component)
        old_name = component.name
        self.component_dict[new_name] = self.component_dict[old_name]
        del self.component_dict[old_name]
        component.name = new_name
        return old_name

    def fit_df(self, *args, **kwargs):
        assert self.active, 'Trying to fit an inactive layer!'
        logging.info('Fitting layer \'{}\'.'.format(self.name))

    def predict_df(self, *args, **kwargs):
        assert self.active, 'Trying to predict with an inactive layer!'
        logging.info('Predicting with layer \'{}\'.'.format(self.name))

    def activate_component(self, component_name='*'):
        for key, component in self.component_dict.items():
            if fnmatch(key, component_name):
                component.active = True

    def deactivate_component(self, component_name='*'):
        for key, component in self.component_dict.items():
            if fnmatch(key, component_name):
                component.active = False

        self.active = False

    def activate(self):
        self.deactivate()

    def deactivate(self):
        self.active = False


class Layer(BaseLayer):
    """Layer providing all functionality to build up the architecture
    and fit/predict.

    Parameters
    ----------
    name : str
        Name of the Layer.

    comment : str, optional (default='')
        Commet for better documentation of the architecture.

    Attributes
    ----------
    name : str
        Name of the layer.

    comment : str
        Comment.

    component_dict : dict
        Dictionary containing all components.

    n_components : int
        Number of components.

    """
    def __init__(self, name, comment=''):
        super(Layer, self).__init__(name, comment=comment)
        self.active = True

    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False

    def fit_df(self, df, kfold, final_model=False):
        """Method to fit all the components.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing all the features.

        kfold : sklean.KFold
            Cross-validation object.

        final_model : bool, optional (defaultFalse)
            Whether the layer is refittet on all samples, after the
            x-validation is run to get scores for all samples.

        Returns
        -------
        new_df : pandas.DataFrame
            Dataframe with the new scores.

        """
        super(Layer, self).fit_df()
        returns = []
        for _, comp in self.component_dict.items():
            returns.extend(comp.returns)
        new_df = pd.DataFrame(index=df.index, columns=returns)
        for train, test in kfold.split(np.empty(df.shape)):
            df_train = df.loc[train, :]
            df_test = df.loc[test, :]
            for key, component in self.component_dict.items():
                if component.active:
                    component.fit_df(df_train)
                    comp_df = component.predict_df(df_test)
                    return_names = component.returns
                    new_df.loc[comp_df.index, return_names] = comp_df
        if final_model:
            for key, component in self.component_dict.items():
                if component.active:
                    component = component.fit_df(df)

                    self.component_dict[component.name] = component
        return new_df

    def predict_df(self, df):
        """Method to get the prediction of all the components.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing all the features.

        Returns
        -------
        new_df : pandas.DataFrame
            Dataframe with the new scores.

        """
        super(Layer, self).predict_df()
        new_df = None
        for key, component in self.component_dict.items():
            if component.active:
                comp_df = component.predict_df(df)
                if new_df is None:
                    new_df = comp_df
                else:
                    new_df = new_df.join(comp_df)
        return new_df


class LayerParallel(Layer):
    """Layer providing all functionality to build up the architecture
    and fit/predict. Provides multiprocessing functionality for
    'fit_df'/'predict_df'.

    Parameters
    ----------
    name : str
        Name of the Layer.

    n_jobs : int
        Max number of parallel processes.

    predict_parallel : bool, optinal (default=False)
        Whether the 'predict_df' funciton should run in parallel.

    fit_parallel : bool, optinal (default=False)
        Whether the 'fit_df' funciton should run in parallel.

    comment : str, optional (default='')
        Commet for better documentation of the architecture.

    Attributes
    ----------
    name : str
        Name of the layer.

    comment : str
        Comment.

    component_dict : dict
        Dictionary containing all components.

    predict_parallel : bool, optinal (default=False)
        Whether the 'predict_df' funciton should run in parallel.

    fit_parallel : bool, optinal (default=False)
        Whether the 'fit_df' funciton should run in parallel.

    n_components : int
        Number of components.

    """

    def __init__(self,
                 name,
                 n_jobs,
                 predict_parallel=False,
                 fit_parallel=True,
                 comment=''):
        super(LayerParallel, self).__init__(name=name,
                                            comment=comment)
        self.n_jobs = n_jobs
        self.fit_parallel = fit_parallel
        self.predict_parallel = predict_parallel

    def fit_predict_single_component(self, component, df_train, df_test):
        df_train = df_train
        df_test = df_test
        component = component.fit_df(df_train)
        return component, component.predict_df(df_test)

    def fit_df(self, df, kfold, final_model=False):
        """Method to fit all the components.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing all the features.

        kfold : sklean.KFold
            Cross-validation object.

        final_model : bool, optional (defaultFalse)
            Whether the layer is refittet on all samples, after the
            x-validation is run to get scores for all samples.

        Returns
        -------
        new_df : pandas.DataFrame
            Dataframe with the new scores.

        """
        super(Layer, self).fit_df()

        if self.fit_parallel:
            returns = []
            for _, comp in self.component_dict.items():
                returns.extend(comp.returns)
            new_df = pd.DataFrame(index=df.index, columns=returns)
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = []
                for train, test in kfold.split(np.empty(df.shape)):
                    df_train = df.loc[train, :]
                    df_test = df.loc[test, :]
                    for key, component in self.component_dict.items():
                        if component.active:
                            sel_att = component.get_needed_features()
                            futures.append(executor.submit(
                                self.fit_predict_single_component,
                                component=component,
                                df_train=df_train.loc[:, sel_att],
                                df_test=df_test.loc[:, sel_att]))
            for future_i in as_completed(futures):
                component, comp_df = future_i.result()
                self.component_dict[component.name] = component
                return_names = component.returns
                new_df.loc[comp_df.index, return_names] = comp_df
            if final_model:
                with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                    for key, component in self.component_dict.items():
                        if component.active:
                            sel_att = component.get_needed_features()
                            component.fit_df(df.loc[:, sel_att])
                    results = wait(futures)
                for i, future_i in enumerate(results.done):
                    component = future_i.result()
                    self.component_dict[component.name] = component
        else:
            new_df = super(LayerParallel, self).fit_df(df,
                                                       kfold=kfold,
                                                       final_model=final_model)
        return new_df

    def predict_df(self, df):
        """Method to get the prediction of all the components.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing all the features.

        Returns
        -------
        new_df : pandas.DataFrame
            Dataframe with the new scores.

        """
        super(Layer, self).predict_df()
        if self.predict_parallel:
            new_df = pd.DataFrame()
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = []
                for key, component in self.component_dict.items():
                    if component.active:
                        sel_att = component.get_needed_features()
                        futures.append(executor.submit(component.predict_df,
                                                       df=df.loc[:, sel_att]))
                results = wait(futures)
            for i, future_i in enumerate(results.done):
                comp_df = future_i.result()
                new_df = new_df.join(comp_df)
        else:
            new_df = super(LayerParallel, self).predict_df(df)
        return new_df
