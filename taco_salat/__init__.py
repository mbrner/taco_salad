#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings

from recipe import Recipe
from component import BaseComponent, Component
from layer import BaseLayer

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

        base_layer = BaseLayer('layer0')
        attribute_component = BaseComponent('attribute')
        label_component = BaseComponent('label')
        weight_component = BaseComponent('weight')
        misc_component = BaseComponent('misc')
        self.add_layer(base_layer)
        super(TacoSalat, self).add_component(layer=base_layer,
                                             component=attribute_component)
        super(TacoSalat, self).add_component(layer=base_layer,
                                             component=label_component)
        super(TacoSalat, self).add_component(layer=base_layer,
                                             component=weight_component)
        super(TacoSalat, self).add_component(layer=base_layer,
                                             component=misc_component)

        if df is not None and roles is not None:
            columns = list(df.columns)
            for name, role in zip(columns, roles):
                if role  == 0:
                    sel_component = attribute_component
                elif role  == 1:
                    sel_component = label_component
                elif role  == 2:
                    sel_component = weight_component
                elif role  == 3:
                    sel_component = misc_component
                else:
                    continue
                self.add_ingredient(unique_name=name,
                                    layer=base_layer,
                                    component=sel_component,
                                    name_layer=name,
                                    role=role)
        else:
            for att in attributes:
                self.add_ingredient(unique_name=att,
                                    layer=base_layer,
                                    component=attribute_component,
                                    name_layer=att,
                                    role=0)
            for label in labels:
                self.add_ingredient(unique_name=att,
                                    layer=base_layer,
                                    component=label_component,
                                    name_layer=att,
                                    role=1)
            for weight_i in weights:
                self.add_ingredient(unique_name=weight_i,
                                    layer=base_layer,
                                    component=weight_component,
                                    name_layer=att,
                                    role=2)
            for misc_i in misc:
                self.add_ingredient(unique_name=att,
                                    layer=base_layer,
                                    component=misc_component,
                                    name_layer=att,
                                    role=3)








    def add_component(self,
                      layer,
                      name,
                      clf,
                      attributes,
                      label,
                      returns,
                      roles=None,
                      weight=None,
                      comment=''):
        if not isinstance(layer, BaseLayer):
            layer = self.get_layer(layer)
        layer_index = self.layer_order.index(layer)

        attribute_names = []
        for att in attributes:
            print(att)
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
        labels = self.get(label)
        assert len(labels) == 1, 'Only 1 ingredient can be used as the label'
        for label_name, ingredient in labels.iterrows():
            att_layer = self.get_layer(ingredient.layer)
            if ingredient.role != 1:
                warnings.warn('{} with role {} used as label!'.format(
                    ingredient.long_name, ingredient.role))

        if weight is not None:
            weight = self.get(weight)
            assert len(weight) == 1, 'Only 1 ingredient useable as the weight'
            for weight_name, ingredient in weight.iterrows():
                att_layer = self.get_layer(ingredient.layer)
                if ingredient.role != 1:
                    warnings.warn('{} with role {} used as weight!'.format(
                        ingredient.long_name, ingredient.role))
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
                              comment=comment)

        super(TacoSalat, self).add_component(layer=layer,
                                             component=component)

        for i, [return_i, role_i] in enumerate(zip(returns, roles)):
            unique_name = self.add_ingredient(unique_name=return_i,
                                              layer=layer,
                                              component=component,
                                              name_layer=return_i,
                                              role=role_i)
            if return_i is None:
                component.returns[i] = unique_name


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



if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier

    iris = load_iris()

    df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                      columns= iris['feature_names'] + ['target'])

    salat = TacoSalat(df, roles=[0, 0, 0, 0, 1])


    first_layer = BaseLayer('FirstClassificationLayer')
    salat.add_layer(first_layer)

    salat.add_component(first_layer,
                        name='RandomForest',
                        clf=RandomForestClassifier(),
                        attributes=['layer0:attribute:*'],
                        label='layer0:label:*',
                        returns=2,
                        weight=None,
                        roles=[0, -1],
                        comment='')

    print(salat.ingredients)

