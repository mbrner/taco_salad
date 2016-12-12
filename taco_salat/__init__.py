#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
import copy

import pandas as pd

from sklearn.model_selection import KFold

from .recipe import Recipe
from .component import BaseComponent, Component
from .layer import BaseLayer, Layer, LayerParallel


class TacoSalat(Recipe):
    """Class provides all funtions to stack methods and perform fit and
    predict. The class has to be initialized with the input features.
    There are 4 diffrent roles for the features:
    - 0 / 'attributes': Features intended for training
    - 1 / 'labels': Features intended to be used as labels
    - 2 / 'weights': Feaures intended to be used as sample weights
    - 3 / 'misc': Features which are carried through all layers without
                a specific intention.

    There are two ways to init the salat. Provided:
    - a Dataframe and an array with the intended roles

    or:

    - arrays with the names of the features for the diffrent roles

    In default a 'layer0' is added. This layer is an object of the
    taco_salat.layer.BaseLayer class. Also four components 'attribute',
    'lablel', 'weight' and 'misc' are added to the 'layer0'. To those
    components the features are added depending on their role. To do
    all steps manually simply init the TacoSalat without any parameters
    and use TacoSalat.add_ingridient to add all the input features.
    The 'rename_component' and 'rename_component' methods can be used
    to change the names of the default layer/components.

    Parameters
    ----------
    kfold : int, optional (default=10)
        Number of cross predict steps for the internal fit.

    df : pandas.DataFrame (n_samples, n_cols) or list (n_cols,) or None
        The dataframe is used to get the names of the columns.
        Those names can be provided directly as a list. In both cases
        a list with the roles for all features.

    roles : array-like (n_cols,) or None
        Array containing the role as int for all features.

    attributes: array-like or None
        Names of all features intended to be used as attributes.

    labels: array-like or None
        Names of all features intended to be used as labels.

    weights: array-like or None
        Names of all features intended to be used as weights.

    misc: array-like or None
        Names of all features without a specific role.

    Attributes
    ----------
    kfold : int
        Number of cross predict steps for the internal fit.

    ingredients : pandas.DataFrame
        Dataframe for the book keeping for all features.

    layer_dict : dict
        Dictionary containing all layers.

    layer_order : list
        List with all layers in the order of execution.

    n_layers : int
        Number of layers

    """
    def __init__(self,
                 kfold=10,
                 df=None,
                 roles=None,
                 attributes=[],
                 labels=[],
                 weights=[],
                 misc=[]):
        self.kfold = kfold

        super(TacoSalat, self).__init__()

        base_layer = BaseLayer('layer0')
        attribute_component = BaseComponent('attribute')
        label_component = BaseComponent('label')
        weight_component = BaseComponent('weight')
        misc_component = BaseComponent('misc')
        super(TacoSalat, self).add_layer(base_layer)
        super(TacoSalat, self).add_component(layer=base_layer,
                                             component=attribute_component)
        super(TacoSalat, self).add_component(layer=base_layer,
                                             component=label_component)
        super(TacoSalat, self).add_component(layer=base_layer,
                                             component=weight_component)
        super(TacoSalat, self).add_component(layer=base_layer,
                                             component=misc_component)

        if df is not None and roles is not None:
            assert isinstance(df, pd.DataFrame) or isinstance(df, list), \
                '\'df\' has to be a Pandas.DataFrame or a list with the ' \
                'column names'
            if isinstance(df, pd.DataFrame):
                columns = list(df.columns)
            else:
                columns = df
            for name, role in zip(columns, roles):
                if role == 0:
                    sel_component = attribute_component
                elif role == 1:
                    sel_component = label_component
                elif role == 2:
                    sel_component = weight_component
                elif role == 3:
                    sel_component = misc_component
                else:
                    continue
                self.add_ingredient(unique_name=name,
                                    layer=base_layer,
                                    component=sel_component,
                                    name_component=name,
                                    role=role)
        else:
            for att in attributes:
                self.add_ingredient(unique_name=att,
                                    layer=base_layer,
                                    component=attribute_component,
                                    name_component=att,
                                    role=0)
            for label_i in labels:
                self.add_ingredient(unique_name=label_i,
                                    layer=base_layer,
                                    component=label_component,
                                    name_component=label_i,
                                    role=1)
            for weight_i in weights:
                self.add_ingredient(unique_name=weight_i,
                                    layer=base_layer,
                                    component=weight_component,
                                    name_component=weight_i,
                                    role=2)
            for misc_i in misc:
                self.add_ingredient(unique_name=misc_i,
                                    layer=base_layer,
                                    component=misc_component,
                                    name_component=misc_i,
                                    role=3)

    def add_layer(self,
                  layer_name=None,
                  n_jobs=1,
                  fit_parallel=True,
                  predict_parallel=False,
                  comment=''):
        """Method to register and add a component to a layer.

        Parameters
        ----------
        layer_name : str
            Name of the Layer

        n_jobs : int, optional (default=1)
            Number of jobs used for the layer. If > 1 the added layer
            is a layer.LayerParallel. The fit_df/predict_df for each
            component is performed in an extra process. Caution if the
            components also have multicore classification/regression
            algorithms.

        fit_parallel : boolean, optional (default=True)
            Only relevant for n_jobs > 1. If True the fit step is done
            parallel.

        predict_parallel : boolean, optional (default=False)
            Only relevant for n_jobs > 1. If True the predict step is done
            parallel.

        comment : sr, optional (default='')
            Comment for better documentation of the architecture.

        Returns
        -------
        layer : taco_salat.layer.Layer or taco_salat.layer.LayerParallel
            The added layer.

        """
        if layer_name is None:
            layer_name = 'layer{}'.format(len(self.layer_order))
        if n_jobs > 1:
            layer = LayerParallel(name=layer_name,
                                  n_jobs=n_jobs,
                                  comment=comment,
                                  fit_parallel=fit_parallel,
                                  predict_parallel=predict_parallel)
        else:
            layer = Layer(name=layer_name,
                          comment=comment)
        super(TacoSalat, self).add_layer(layer)
        return layer

    def add_component(self,
                      layer,
                      name,
                      clf,
                      attributes,
                      label,
                      returns,
                      roles=None,
                      weight=None,
                      fit_func='fit',
                      predict_func='predict_proba',
                      comment=''):
        """Method to register and add a component to a component.

        Parameters
        ----------
        layer : str or taco_salat.layer.Layer
            Name of the Layer

        name : str
            Name of the component

        attributes : list of str
            List with all features used as attributes.

        label : str
            Name of the feature used as the label.

        returns : int or list of str
            Number of features that will be returnd by the 'clf' or names

        roles : None or array-like (ints) shape=(n_returns,), optional
            If None all returns are treated as attributes (role=0). Or
            the different roles can be defined as an array of ints. To
            ignore returned values of the clf just use a
            role != [0, 1, 2, 3]

        weight : None or str
            None for no sample weights or name of the sample weight.

        fit_func : str, optional (default='fit')
            Name of the function, that should be called in the 'fit_df'
            function.

        predict_func: str, optional (default='predict_proba')
            Name of the function, that should be called in The
            'predict_df' function.

        comment : sr, optional (default='')
            Comment for better documentation of the architecture.

        Returns
        -------

        component : taco_salat.component.Component
            The added component.

        """
        if not isinstance(layer, BaseLayer):
            layer = self.get_layer(layer)
        layer_index = self.layer_order.index(layer)

        attribute_names = []
        dependencies = []
        for att in attributes:
            selected_att = self.get(att)
            for att_name, ingredient in selected_att.iterrows():
                att_layer = self.get_layer(ingredient.layer)
                layer_index_att = self.layer_order.index(att_layer)
                assert layer_index > layer_index_att, \
                    '{} not from a lower layer!'.format(ingredient.long_name)
                if ingredient.role != 0:
                    warnings.warn('{} with role {} used as attribute!'.format(
                        ingredient.long_name, ingredient.role))
                attribute_names.append(att_name)
                dependencies.append(att_name)
        labels = self.get(label)
        assert len(labels) == 1, 'Only 1 ingredient can be used as the label'
        for label_name, ingredient in labels.iterrows():
            att_layer = self.get_layer(ingredient.layer)
            if ingredient.role != 1:
                warnings.warn('{} with role {} used as label!'.format(
                    ingredient.long_name, ingredient.role))
            dependencies.append(label_name)
        if weight is not None:
            weight = self.get(weight)
            assert len(weight) == 1, 'Only 1 ingredient useable as the weight'
            for weight_name, ingredient in weight.iterrows():
                att_layer = self.get_layer(ingredient.layer)
                if ingredient.role != 2:
                    warnings.warn('{} with role {} used as weight!'.format(
                        ingredient.long_name, ingredient.role))
                dependencies.append(weight_name)
        else:
            weight_name = None

        if isinstance(returns, int):
            returns = [None] * returns
        if roles is None:
            roles = [0] * len(returns)

        component = Component(name,
                              clf=clf,
                              attributes=attribute_names,
                              label=label_name,
                              returns=returns,
                              weight=weight_name,
                              fit_func=fit_func,
                              predict_func=predict_func,
                              comment=comment)

        super(TacoSalat, self).add_component(layer=layer,
                                             component=component)
        for i, [return_i, role_i] in enumerate(zip(returns, roles)):
            unique_name = self.add_ingredient(unique_name=return_i,
                                              layer=layer,
                                              component=component,
                                              name_component=return_i,
                                              role=role_i)
            if unique_name is not None:
                component.returns[i] = unique_name
        self.set_dependencies(dependencies, layer=layer, component=component)
        return component

    def fit_df(self, df, clear_df=True, final_mode=None):
        """Method to fit all components. To keep train/test data
        disjunct all the time and to train all layers with the maximal
        possible amout of samples. Each component is trained in a
        cross validation to get a prediction for each sample.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the complete data. All features
            registered in the lowest layer (default: 'layer0') have to
            be a columns in the dataframe.

        clear_df : boolean, optional (default=True)
            The output of all layers is added to the dataframe during
            the fit process. If True all added columns are removed at
            the end.

        final_mode : ['last'/True, 'all', None/False]
            The last layer can be trained on all samples, because there
            isn't a component depending on the output. If 'True' the
            last layer is trained with all samples. If 'all' all layers
            are retrained after all samples were scored in the
            x-validation.
            Note: If final_mode is 'False' and clear_df is
            'False' the df has the out of the bag score for the complete
            salat.

        Returns
        -------

        df : pandas.Dataframe
            The input dataframe with the out of the bag scores if
            'clear_df' is False.

        """
        df_input_cols = df.columns
        kf = KFold(n_split=self.kfold, shuffle=True)
        for i, layer in enumerate(self.layer_order):
            if layer.active:
                if final_mode.lower() == 'all':
                    final_model = True
                elif final_mode.lower() == 'last' or final_mode:
                    if layer == self.layer_order[-1]:
                        final_model = True
                    else:
                        final_model = False
                else:
                    final_model = False
                layer_df = layer.fit_df(df,
                                        kfold=kf,
                                        final_model=final_model)
                if isinstance(layer_df, pd.DataFrame):
                    layer_entries = self.get('{}.*'.format(layer.name))
                    layer_obs = [name for name, _ in layer_entries.iterrows()]
                    layer_df = layer_df.loc[:, layer_obs]
                    df.join(layer_df)
        if clear_df:
            df_final_cols = df.columns
            drop_cols = [col for col in df_final_cols
                         if col not in df_input_cols]
            df = df.drop(drop_cols)

    def predict_df(self, df, clear_df=False):
        """Method to apply the complete salat to a dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the complete data. All features
            registered in the lowest layer (default: 'layer0') have to
            be a columns in the dataframe.

        clear_df : boolean, optional (default=True)
            The output of all layers is added to the dataframe during
            the fit process. If True all added columns are removed at
            the end except the out of the last layer.

        Returns
        -------

        df : pandas.Dataframe
            The input dataframe with the scores for all layers
            (clear_df=False) or only the last layers scores
            (clear_df=True).

        """

        df_input_cols = df.columns
        for layer in self.layer_order:
            if layer.active:
                layer_df = layer.predict_df(df)
                if isinstance(layer_df, pd.DataFrame):
                    layer_entries = self.get('{}:*'.format(layer.name))
                    layer_obs = [name for name, _ in layer_entries.iterrows()]
                    layer_df = layer_df.loc[:, layer_obs]
                    df = df.join(layer_df)
        if clear_df:
            df_final_cols = df.columns
            drop_cols = [col for col in df_final_cols
                         if col not in df_input_cols not in layer_obs]
            df = df.drop(drop_cols)
        return df

    def fit_component(self, df, long_name, clear_df=True, final_mode=None):
        def wrapped_fit_call():
            return self.fit_df(df=df,
                               clear_df=clear_df,
                               final_mode=final_mode)

        return self.__run_func_component__(wrapped_fit_call, long_name)

    def predict_component(self, df, long_name, clear_df=True):
        def wrapped_predict_call():
            return self.predict_df(df=df,
                                   clear_df=clear_df)

        return self.__run_func_component__(wrapped_predict_call, long_name)
