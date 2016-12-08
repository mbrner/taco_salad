#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from fnmatch import fnmatch
from component import BaseComponent
from layer import BaseLayer


class Recipe(object):
    """Base Class providing an interface to stack classification layers.

    Parameters
    ----------
    label_factorty: instance of utensils.LabelFactory
        Instance of LabelFactory used to generate Label from the Data.

    Attributes
    ----------
    ingredients : pandas.DataFrame
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
        assert layer.name not in self.layer_dict.keys(), \
            '{} already exists'.format(name)
        self.layer_dict[layer.name] = layer
        self.n_layers += 1
        self.layer_order.append(layer)
        return self.layer_dict[layer.name]

    def get_layer(self, name):
        if isinstance(name, int):
            name = 'layer{}'.format(name)
        assert name in self.layer_dict.keys(), 'Unkown Layer {}'.format(name)
        return self.layer_dict[name]

    def rename_layer(self, layer, new_name):
        if not isinstance(layer, BaseLayer):
            layer = self.get_layer(layer)
        assert new_name not in self.layer_dict.keys(), \
            '{} already exists'.format(new_name)
        old_name = layer.name
        self.layer_dict[new_name] = self.layer_dict[old_name]

        del self.layer_dict[old_name]

        layer.name = new_name
        self.rename_ingredients('layer', old_name, new_name)

    def add_component(self, layer, component):
        if not isinstance(layer, BaseLayer):
            layer = self.get_layer(layer)
        layer.add_component(component)
        return layer

    def get_component(self, layer, name):
        if not isinstance(layer, BaseLayer):
            layer = self.get_layer(name)
        return layer[name]

    def rename_component(self, layer, component, new_name):
        if not isinstance(layer, BaseLayer):
            layer = self.get_layer(layer)
        old_name = layer.rename_component(component, new_name)

        self.rename_ingredients('component', old_name, new_name)

    def __generate_long_name__(self, layer, component, name_layer):
        return '{}:{}:{}'.format(layer, component, name_layer)

    def add_ingredient(self,
                       unique_name=None,
                       layer='layer0',
                       component='input',
                       name_layer=None,
                       role=0):
        """Method to add all parameters.

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
                3 : Only used for labeling
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

    def get(self, att, role=None):
        df = self.ingredients
        if isinstance(att, int):
            return df.iloc[[att]]
        elif att in df.index:
            return df[[att]]
        else:
            idx = df.long_name.apply(fnmatch, pat=att)
            return df[idx]

    def rename_ingredients(self, col, old_name, new_name):
        idx = self.ingredients.loc[:, col].values == old_name
        self.ingredients.loc[idx, col] = new_name
        for name, entry in self.ingredients[idx].iterrows():
            long_name = self.__generate_long_name__(entry.layer,
                                                    entry.component,
                                                    entry.name_layer)
            self.ingredients.loc[name, 'long_name'] = long_name




if __name__ == '__main__':
    from IPython import embed
    r = Recipe()


    family_layer = BaseLayer('Familie')
    r.add_layer(family_layer)
    eltern = BaseComponent('Eltern')
    r.add_component(family_layer, eltern)
    geschwister = BaseComponent('Geschwister')
    r.add_component(family_layer, geschwister)
    r.add_ingredient('Horst', family_layer, eltern, 'Vater')
    r.add_ingredient('Marie', 'Familie', 'Eltern', 'Mutter')
    r.add_ingredient('Anna', 'Familie', geschwister, 'Schwester')
    r.add_ingredient('Lukas', family_layer, 'Geschwister', 'Bruder')

    friends_layer = BaseLayer('Friends')
    r.add_layer(friends_layer)

    inner_circle_friends = BaseComponent('inner_circle')
    inner_circle_friends = r.add_component(friends_layer, inner_circle_friends)
    r.add_ingredient('Hanne', friends_layer, inner_circle_friends, 'Freundin')
    r.add_ingredient('Peter', friends_layer, inner_circle_friends)
    r.add_ingredient('Hansi', friends_layer, inner_circle_friends)

    print(r.ingredients)





