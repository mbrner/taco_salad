#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from .component import Component


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
