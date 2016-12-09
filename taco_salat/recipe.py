#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from fnmatch import fnmatch
from component import BaseComponent
from layer import BaseLayer


class Recipe(object):
    """Base Class providing an interface to stack classification layers.

    Attributes
    ----------
    ingredients : pandas.DataFrame
        Containing the detailed ingredient list.
        - 'long_name' : <layer>:<component>:<name_layer>
        - 'layer' : Name of the layer containg the ingredient.
        - 'component' : Name of the component containing the ingredient
        - 'name_layer' : Name of the ingredient inside the component
        - 'role' : 0, 1, 2 or 3 depending on the role (see add_ingredient)
        The index of the dataframe is the 'unique_name' of the
        ingredients.
    """
    def __init__(self):
        self.ingredients = pd.DataFrame(columns=['long_name',
                                                 'layer',
                                                 'component',
                                                 'name_layer',
                                                 'role'])
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

    def __generate_long_name__(self, layer, component, name_layer):
        return '{}:{}:{}'.format(layer, component, name_layer)

    def add_ingredient(self,
                       unique_name=None,
                       layer='layer0',
                       component='input',
                       name_layer=None,
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

        name_layer : str
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
            unique_name = str(len(df))
        assert unique_name not in df.index, \
            '{} already exists'.format(unique_name)

        if not isinstance(layer, BaseLayer):
            layer = self.get_layer(layer)

        if not isinstance(component, BaseComponent):
            component = self.get_component(layer, component)

        if name_layer is None:
            name_layer = str(len(df))

        layer_names = df.name_layer[df.layer == layer.name]
        assert name_layer not in layer_names, \
            '{} already exists in layer {}'.format(unique_name, layer)

        long_name = self.__generate_long_name__(layer.name,
                                                component.name,
                                                name_layer)
        df.loc[unique_name, :] = [long_name,
                                  layer.name,
                                  component.name,
                                  name_layer,
                                  role]
        return unique_name

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
            return df[[att]]
        else:
            idx = df.long_name.apply(fnmatch, pat=att)
            return df[idx]

    def __rename_ingredients__(self, col, old_name, new_name):
        idx = self.ingredients.loc[:, col].values == old_name
        self.ingredients.loc[idx, col] = new_name
        for name, entry in self.ingredients[idx].iterrows():
            long_name = self.__generate_long_name__(entry.layer,
                                                    entry.component,
                                                    entry.name_layer)
            self.ingredients.loc[name, 'long_name'] = long_name

    def get_ingredient_list(self, long_name=False):
        """Get all ingredient as list.

        Parameters
        ----------
        long_name : boolean, optinal (default=False)
            If 'False' the 'unique_names' are returned.
            If 'True' the 'long_names' are returned:
                <layer_name>:<component_name>:<name_layer>

        Returns
        -------
        ingredients : list
            List of the names.
        """
        if long_name:
            return list(self.ingredients.loc[:, 'long_name'])
        else:
            return list(self.ingredients.index)
