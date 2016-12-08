#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from fnmatch import fnmatch

class Component(object):
    def __init__(self, name, comment='', **kwargs):
        self.name = name
        self.comment = comment
        self.component_dict = {}


class Layer(object):
    def __init__(self, name, comment=''):
        self.name = name
        self.comment = comment
        self.component_dict = {}
        self.n_components = 0

    def add_component(self, name, comment='', **kwargs):
        assert name not in self.component_dict.keys(), \
            '{} already in layer {}'.format(name, self.name)
        self.n_components += 1
        self.component_dict[name] = Component(name, comment)
        return self.component_dict[name]

    def get_component(self, name):
        return self.component_dict[name]

    def __getitem__(self, name):
        return self.get_component(name)

    def rename_component(self, component, new_name):
        if not isinstance(component, Component):
            component = self.get_component(component)
        old_name = component.name
        self.component_dict[new_name] = self.component_dict[old_name]
        del self.component_dict[old_name]
        component.name = new_name
        return old_name



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
        self.n_layers = 0
        self.base_layer = self.add_layer(
            name='layer0',
            comment='Layer for all initial ingredients.')
        self.base_layer.add_component(
            name='input',
            comment='Present in the input.')

    def add_layer(self, name=None, comment=''):
        if name is None:
            name = 'layer{}'.format(self.n_layers)
        assert name not in self.layer_dict.keys(), \
            '{} already exists'.format(name)
        self.layer_dict[name] = Layer(name=name,
                                      comment=comment)
        self.n_layers += 1
        return self.layer_dict[name]

    def get_layer(self, name):
        if isinstance(name, int):
            name = 'layer{}'.format(name)
        assert name in self.layer_dict.keys(), 'Unkown Layer {}'.format(name)
        return self.layer_dict[name]

    def rename_layer(self, layer, new_name):
        if not isinstance(layer, Layer):
            layer = self.get_layer(layer)
        assert new_name not in self.layer_dict.keys(), \
            '{} already exists'.format(new_name)
        old_name = layer.name
        self.layer_dict[new_name] = self.layer_dict[old_name]
        del self.layer_dict[old_name]

        layer.name = new_name
        self.rename_ingredients('layer', old_name, new_name)



    def add_component(self, layer, name=None, comment='', **kwargs):
        if not isinstance(layer, Layer):
            layer = self.get_layer(layer)
        if name is None:
            layer_id = layer.n_components
            name = 'component{}'.format(layer_id)
        return layer.add_component(name=name)

    def get_component(self, layer, name):
        if not isinstance(layer, Layer):
            layer = self.get_layer(name)
        return layer[name]

    def rename_component(self, layer, component, new_name):
        if not isinstance(layer, Layer):
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
        df = self.ingredients
        if unique_name is None:
            unique_name = str(len(df))
        assert unique_name not in df.index, \
            '{} already exists'.format(unique_name)

        if not isinstance(layer, Layer):
            layer = self.get_layer(layer)

        if not isinstance(component, Component):
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

    def get(self, att):
        df = self.ingredients
        if isinstance(att, int):
            return df.role.iloc[[att]]
        elif att in df.index:
            return df.role.loc[[att]]
        else:
            idx = df.long_name.apply(fnmatch, pat=att)
            return df.role[idx]

    def rename_ingredients(self, col, old_name, new_name):
        idx = self.ingredients.loc[:, col].values == old_name
        self.ingredients.loc[idx, col] = new_name
        for name, entry in self.ingredients[idx].iterrows():
            long_name = self.__generate_long_name__(entry.layer,
                                                    entry.component,
                                                    entry.name_layer)
            self.ingredients.loc[name, 'long_name'] = long_name




if __name__ == '__main__':
    r = Recipe()
    family_layer = r.add_layer('Familie')
    eltern = r.add_component(family_layer, 'Eltern')
    geschwister = r.add_component(family_layer, 'Geschwister')
    r.add_ingredient('Horst', family_layer, eltern, 'Vater')
    r.add_ingredient('Marie', 'Familie', 'Eltern', 'Mutter')
    r.add_ingredient('Anna', 'Familie', geschwister, 'Schwester')
    r.add_ingredient('Lukas', family_layer, 'Geschwister', 'Bruder')

    friends_layer = r.add_layer('friends')
    inner_circle_friends = r.add_component(friends_layer, 'inner_circle')
    r.add_ingredient('Hanne', friends_layer, inner_circle_friends, 'Freundin')
    r.add_ingredient('Peter', friends_layer, inner_circle_friends)
    r.add_ingredient('Hansi', friends_layer, inner_circle_friends)

    kollegen_layer = r.add_layer()
    uni = r.add_component(kollegen_layer)

    r.add_ingredient('Wolfgang', kollegen_layer, uni)
    print(r.ingredients)
    r.rename_layer(kollegen_layer, 'Kollegen')
    print('.............')
    r.rename_component(kollegen_layer, uni, 'Uni')
    print(r.ingredients)
    print(r.get('Familie*'))




