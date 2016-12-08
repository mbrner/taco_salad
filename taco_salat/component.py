#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd

class Component(object):
    def __init__(self,
                 name,
                 clf,
                 attributes,
                 label,
                 weight=None,
                 comment=''):
        self.name = name
        self.comment = comment
        self.component_dict = {}
