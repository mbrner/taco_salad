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
        self.clf = None
        self.attributes = None
        self.label = None
        self.returns = None
        self.weigt = None
