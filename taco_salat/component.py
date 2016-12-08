#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd

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
                 comment=''):
        super(Component, self).__init__(name=name,
                                        comment=comment)
        self.clf = clf
        self.attributes = attributes
        self.label = label
        self.returns = returns
        self.weight = weight
