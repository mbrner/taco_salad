#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import logging

from fnmatch import fnmatch

import pandas as pd

from .component import BaseComponent
from .layer import BaseLayer


class Recipe(object):
    """Base Class providing an interface to stack classification layers.

    Attributes
    ----------
    ingredients : pandas.DataFrame
        Containing the detailed ingredient list.
        - 'long_name' : <layer>:<component>:<name_component>
        - 'layer' : Name of the layer containg the ingredient.
        - 'component' : Name of the component containing the ingredient
        - 'name_component' : Name of the ingredient inside the component
        - 'role' : 0, 1, 2 or 3 depending on the role (see add_ingredient)
        The index of the dataframe is the 'unique_name' of the
        ingredients.
    """
    def __init__(self):
        self.ingredients = pd.DataFrame(columns=['long_name',
                                                 'layer',
                                                 'component',
                                                 'name_component',
                                                 'role'])
        self.dependencies = pd.DataFrame()
        self.layer_dict = {}
        self.layer_order = []
        self.n_layers = 0

    def add_layer(self, layer):
        """ Add a layer to the recipe.

       Parameters
        -------
        layer : taco_salat.layer.Layer or taco_salat.layer.LayerParallel
            Layer object.

        Returns
        -------
        layer : layer.Layer/BaseLayer/LayerParallel...
            Added layer.

        """

        if layer.name in self.layer_dict.keys():
            raise KeyError('{} already exists'.format(layer.name))
        self.layer_dict[layer.name] = layer
        self.n_layers += 1
        self.layer_order.append(layer)
        logging.info('Layer \'{}\' added to the Recipe ({}-th layer).'.format(
                     layer.name, self.n_layers - 1))
        return self.layer_dict[layer.name]

    def get_layer(self, name):
        """Get layer matching the name.
        If layer is unkown an KeyError is raised.

        Parameters
        ----------
        name : str or int
            If string 'name' have to be the complete name of the layer.
            If int the layer has to be 'layer<name>'. This naming is
            used when layers are added without providing a name.

        Returns
        -------
        layer : layer.Layer/BaseLayer/LayerParallel...
            Matching layer.
        """
        if isinstance(name, int):
            name = 'layer{}'.format(name)
        return self.layer_dict[name]

    def rename_layer(self, layer, new_name):
        """Rename a component. Also the long_names are adjusted.

        Parameters
        ----------
        layer : str or int or layer.Layer/BaseLayer/LayerParallel...
            Name of the layer (see get_layer()) or the layer object.

        new_name : str
            New name of the component.
        """
        if not isinstance(layer, BaseLayer):
            layer = self.get_layer(layer)
        if new_name in self.layer_dict.keys():
            raise KeyError('{} already exists'.format(new_name))
        old_name = layer.name
        self.layer_dict[new_name] = self.layer_dict[old_name]

        del self.layer_dict[old_name]

        layer.name = new_name
        self.__rename_ingredients__('layer', old_name, new_name)

    def add_component(self, layer, component):
        """Add component to a layer.
        If layer is unkown an KeyError is raised.

        Parameters
        ----------
        layer : str or int or layer.Layer/BaseLayer/LayerParallel...
            Name of the layer (see get_layer()) or the layer object.

        component : component.Component/BaseComponent
            Component object

        Returns
        -------
        component : component.Component/BaseComponent
            Added component.
        """
        if not isinstance(layer, BaseLayer):
            layer = self.get_layer(layer)
        layer.add_component(component)
        logging.info('Component \'{}\' added to \'{}\'.'.format(
                     component.name, layer.name))
        return layer

    def get_component(self, layer, name):
        """Get the component from a layer.
        If layer/component is unkown an KeyError is raised.

        Parameters
        ----------
        layer : str or int or layer.Layer/BaseLayer/LayerParallel...
            Name of the layer (see get_layer()) or the layer object.

        name : str
            Name of the component.

        dependencies : list of str
            Names of attributes needed for this component.

        Returns
        -------
        component : component.Component/BaseComponent
            Matching component.

        """
        if not isinstance(layer, BaseLayer):
            layer = self.get_layer(name)
        return layer[name]

    def rename_component(self, layer, component, new_name):
        """Rename a component. Also the long_names are adjusted.

        Parameters
        ----------
        layer : str or int or layer.Layer/BaseLayer/LayerParallel...
            Name of the layer (see get_layer()) or the layer object.

        component : str or component.Component/BaseComponent/...
            Name of the component or component object.

        new_name : str
            New name of the component.

        """
        if not isinstance(layer, BaseLayer):
            layer = self.get_layer(layer)
        old_name = layer.rename_component(component, new_name)

        self.__rename_ingredients__('component', old_name, new_name)

    def get_layer_and_component(self,
                                long_name=None,
                                layer=None,
                                component=None):
        if long_name is not None:
            splitted_name = long_name.split(':')
            layer_name = splitted_name[0]
            comp_name = splitted_name[1]
            layer = self.get_layer(layer_name)
            component = self.get_component(layer, comp_name)
        elif layer is not None and component is not None:
            if not isinstance(layer, BaseLayer):
                layer = self.get_layer(layer)
            if not isinstance(component, BaseComponent):
                component = self.get_component(layer, component)
        return layer, component

    def __generate_long_name__(self, layer, component, name_component):
        return '{}:{}:{}'.format(layer, component, name_component)

    def add_ingredient(self,
                       unique_name=None,
                       layer='layer0',
                       component='input',
                       name_component=None,
                       role=0):
        """Method to register a ingredient.

        Parameters
        ----------
        unique_name : str
            Unique name for the ingredient.

        layer : int
            Index of the layer

        component : int
            Index of the component.

        name_component : str
            Name inside the layer must not be globally unique,
            but inside the layer. Used to reference as layer:component:name.

        role : int
            Role of the ingredient:
                0 : Attribute
                1 : Label
                2 : Weight
                3 : Only used for labeling (misc)
        """
        if role not in [0, 1, 2, 3]:
            return None
        df = self.ingredients
        if unique_name is None:
            unique_name = 'score_{}'.format(len(df))
        assert unique_name not in df.index, \
            '{} already exists'.format(unique_name)

        if not isinstance(layer, BaseLayer):
            layer = self.get_layer(layer)

        if not isinstance(component, BaseComponent):
            component = self.get_component(layer, component)

        if name_component is None:
            name_component = str(len(df))

        component_features = df.name_component[df.component == component.name]
        assert name_component not in component_features, \
            '{} already exists in layer {}'.format(unique_name, layer)

        long_name = self.__generate_long_name__(layer.name,
                                                component.name,
                                                name_component)
        df.loc[unique_name, 'long_name'] = long_name
        df.loc[unique_name, 'layer'] = layer.name
        df.loc[unique_name, 'component'] = component.name
        df.loc[unique_name, 'name_component'] = name_component
        df.loc[unique_name, 'role'] = role
        self.dependencies.loc[unique_name, :] = False
        return unique_name

    def set_dependencies(self,
                         dependencies,
                         att_name=None,
                         layer=None,
                         component=None,
                         component_long_name=None):
        if att_name is not None:
            attribute = self.get(att_name)
            dependency_col = '{}:{}'.format(attribute.iloc[0].layer,
                                            attribute.iloc[0].component)
        elif layer is not None and component is not None:
            layer, component = self.get_layer_and_component(
                long_name=component_long_name,
                layer=layer,
                component=component)
            dependency_col = '{}:{}'.format(layer.name, component.name)
        relevant_features = [self.get(dep) for dep in dependencies]
        relevant_features = pd.concat(relevant_features, axis=0)
        relevant_features = relevant_features.drop_duplicates().index
        self.dependencies.loc[:, dependency_col] = False
        self.dependencies.loc[relevant_features, dependency_col] = True

    def __resolve_long_name__(self, long_name):
        splitted_name = long_name.split(':')
        if len(splitted_name) == 1:
            layer = self.get_layer(long_name)
            unique_name = None
            component = None
        elif len(splitted_name) == 2:
            layer, component = self.get_layer_and_component(long_name)
            unique_name = None
        elif len(splitted_name) == 3:
            layer, component = self.get_layer_and_component(long_name)
            unique_name = self.get(long_name)
            if len(unique_name) == 0:
                raise KeyError('Feature part of long name unkown!')
            elif len(unique_name) > 1:
                raise KeyError('Feature part of long name not unique!')
            else:
                unique_name = unique_name.index.values[0]
        return layer, component, unique_name

    def resolve_dependencies(self, long_name):
        if long_name not in self.dependencies.columns:
            raise KeyError('{} not found!'.format(long_name))
        component_dependencies = self.dependencies.loc[:, long_name]
        dependencies = set()
        for unique_name, used in component_dependencies.iteritems():
            if used:
                layer = self.ingredients.loc[unique_name, 'layer']
                component = self.ingredients.loc[unique_name, 'component']
                if layer == 'layer0':
                    dependencies.add(unique_name)
                else:
                    dependencies.add(unique_name)
                    component_name = '{}:{}'.format(layer, component)
                    additions = self.resolve_dependencies(component_name)
                    dependencies = dependencies.union(additions)
        return dependencies

    def get(self, att):
        """Methods to get the part of the ingredients DataFrame matching
        the 'att' string.

        Parameters
        ----------
        att : str or int
            String to 'search' ingredients. If int the ath-th ingredient
            is returnd. If 'att' is a string all ingredients with a
            matching 'unique_namer' or 'long_name' are returned.
            For a matching ingredient the 'unique_name'/'long_name' and
            att have to be either equal or matching in the 'fnmatch'
            logic. Looks for equal strings or '*' are added and treated
            like wildcards (bash style.

        Returns
        -------
        df : pandas.DataFrame
            Part of the ingredients dataframe matching 'att'.
        """
        df = self.ingredients
        if isinstance(att, int):
            return df.iloc[[att]]
        elif att in df.index:
            return df.loc[[att]]
        else:
            idx = df.long_name.apply(fnmatch, pat=att)
            return df[idx]

    def __rename_ingredients__(self, col, old_name, new_name):
        idx = self.ingredients.loc[:, col].values == old_name
        self.ingredients.loc[idx, col] = new_name
        for name, entry in self.ingredients[idx].iterrows():
            long_name = self.__generate_long_name__(entry.layer,
                                                    entry.component,
                                                    entry.name_component)
            self.ingredients.loc[name, 'long_name'] = long_name

    def get_ingredient_list(self, long_name=False):
        """Get all ingredient as list.

        Parameters
        ----------
        long_name : boolean, optinal (default=False)
            If 'False' the 'unique_names' are returned.
            If 'True' the 'long_names' are returned:
                <layer_name>:<component_name>:<name_component>

        Returns
        -------
        ingredients : list
            List of the names.
        """
        if long_name:
            return list(self.ingredients.loc[:, 'long_name'])
        else:
            return list(self.ingredients.index)

    def get_activity_state(self):
        components = self.ingredients.loc[:, 'component']
        layers = self.ingredients.loc[:, 'layer']
        component_long_names = set(['{}:{}'.format(lay, comp)
                                    for lay, comp in zip(layers.values,
                                                         components.values)])
        state_layers = pd.Series(True, index=set(layers.values))
        state_components = pd.Series(True, index=component_long_names)
        for layer_i in state_layers.index:
            state_layers.loc[layer_i] = self.layer_dict[layer_i].active

        for long_i in component_long_names:
            layer_i, comp_i = long_i.split(':')
            state = self.layer_dict[layer_i][comp_i].active
            state_components.loc[long_i] = state
        return state_layers, state_components

    def set_activity_state(self, state_layers=None, state_components=None):
        if isinstance(state_layers, tuple):
            state_components = state_layers[1]
            state_layers = state_layers[0]
        if state_layers is not None:
            for layer_i, state in state_layers.iteritems():
                if state:
                    self.layer_dict[layer_i].activate()
                else:
                    self.layer_dict[layer_i].deactivate()

        if state_components is not None:
            for long_i, state in state_components.iteritems():
                layer_i, comp_i = long_i.split(':')
                if state:
                    self.layer_dict[layer_i][comp_i].activate()
                else:
                    self.layer_dict[layer_i][comp_i].deactivate()

    def __run_func_component__(self, func, long_name):
        dependencies = self.resolve_dependencies(long_name)
        dependency_dict = {}

        for unique_name in dependencies:
            layer = self.ingredients.loc[unique_name, 'layer']
            component = self.ingredients.loc[unique_name, 'component']
            if layer not in dependency_dict.keys():
                dependency_dict[layer] = set()
            dependency_dict[layer].add(component)

        prev_state_layer, prev_state_comp = self.get_activity_state()
        requested_state_layer = pd.Series(False, index=prev_state_layer.index)
        requested_state_comp = pd.Series(False, index=prev_state_comp.index)
        for layer, components in dependency_dict.items():
            requested_state_layer.loc[layer] = True
            for comp in components:
                long_name_i = '{}:{}'.format(layer, comp)
                requested_state_comp.loc[long_name_i] = True

        requested_state_layer.loc[long_name.split(':')[0]] = True
        requested_state_comp.loc[long_name] = True
        self.set_activity_state(state_layers=requested_state_layer,
                                state_components=requested_state_comp)
        new_layer_state, new_comp_state = self.get_activity_state()
        returned_values = func()
        self.set_activity_state(state_layers=prev_state_layer,
                                state_components=prev_state_comp)
        return returned_values

    def deactivate_component(self, long_name=None, layer=None, component=None):
        layer, component = self.get_layer_and_component(long_name=long_name,
                                                        layer=layer,
                                                        component=component)
        component_returns = component.returns
        component.deactivate()
        for unique_name in component_returns:
            dependcy_series = self.dependencies.loc[unique_name, :]
            for long_name_i, used in dependcy_series.iteritems():
                if used:
                    self.deactivate_component(long_name=long_name_i)

    def activate_component(self, long_name=None, layer=None, component=None):
        layer, component = self.get_layer_and_component(long_name=long_name,
                                                        layer=layer,
                                                        component=component)
        component_returns = component.returns
        component.activate()
        for unique_name in component_returns:
            dependcy_series = self.dependencies.loc[unique_name, :]
            for long_name_i, used in dependcy_series.iteritems():
                if used:
                    self.activate_component(long_name=long_name_i)

    def deactivate_layer(self, layer, first=True):
        if not isinstance(layer, BaseLayer):
            layer = self.get_layer(layer)
        if first:
            layer.deactivate()
        unique_names = self.get('{}:*'.format(layer.name))
        for unique_name in unique_names.index:
            dependcy_series = self.dependencies.loc[unique_name, :]
            for long_name_i, used in dependcy_series.iteritems():
                if used:
                    self.deactivate_component(long_name=long_name_i)

    def activate_layer(self, layer):
        if not isinstance(layer, BaseLayer):
            layer = self.get_layer(layer)
        layer.deactivate()
        unique_names = self.get('{}:*'.format(layer.name))
        for unique_name in unique_names.index:
            dependcy_series = self.dependencies.loc[unique_name, :]
            for long_name_i, used in dependcy_series.iteritems():
                if used:
                    self.deactivate_component(long_name=long_name_i)
