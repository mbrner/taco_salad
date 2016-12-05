#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


class TwoWayDict(dict):
    """Dict to connected entries in both directions. Keys to content and wise
    versa. So there is no difference between keys and content

    Parameters
    ----------
    one_way_dict: dict
        Normal dict which gets connected in both directions
    """
    def __init__(self, one_way_dict):
        for key, value in one_way_dict.iteritems():
            self.__setitem__(key, value)

    def __setitem__(self, key, value):
        # Remove any previous connections with these values
        if key in self:
            del self[key]
        if value in self:
            del self[value]
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

    def __delitem__(self, key):
        dict.__delitem__(self, self[key])
        dict.__delitem__(self, key)

    def __len__(self):
        """Returns the number of connections"""
        return dict.__len__(self) // 2


class LabelFactory:
    """Base Class providing an interface to stack classification layers.

    Attributes
    ----------
        types: List of TypeEntry
            List storing all registered types.
    """
    def __init__(self):
        self.types = []

    class TypeEntry:
        """Internal class of LabelFactory used to stores the different Types.

        Attributes
        ----------
            id: int
                Identifier Number of the Label.

            function: function
                Callable that returns True if an entries is of the
                certain type.

            name: str or None, optional (default=None)
                Name of the label, if None the id is also used as the
                name.
    """
        def __init__(self, type_id, function, name=None, weighted=False):
            self.id = type_id
            self.function = function
            self.weighted = weighted
            if name is None:
                self.name = str(id)
            else:
                self.name = name

        def __eq__(self, counterpart):
            if isinstance(counterpart, self):
                return self.id == counterpart.id
            elif isinstance(counterpart, int):
                return self.id == counterpart
            elif isinstance(counterpart, str):
                return self.name == counterpart

        def apply(self, df):
            is_type = self.function(df)
            idx = df.loc[is_type].index
            return pd.Series(self.id, index=idx)

        def __call__(self, df):
            self.apply(df)

    def add_type(self, decision_function, name=None):
        type_id = 2**(len(self.types) + 1)
        type_entry = self.TypeEntry(id=type_id,
                                    function=decision_function,
                                    name=name)
        self.types.append(type_entry)
        return type_entry

    def set_types(self, df, type_col='Type'):
        df[type_col] = pd.Series(0, index=df.index, dtype=int)
        for t in self.types:
            df[type_col] = t(df)
        return df

    def __get_type_from_str__(self, type_str):
        try:
            t = self.types[self.types.index(type_str)]
        except ValueError:
            pass
        try:
            type_int = int(type_str)
        except ValueError:
            raise ValueError('{} is no valid type'.format(type_str))
        else:
            try:
                t = self.types[self.types.index(type_int)]
            except ValueError:
                t_list = []
                binary_code = bin(type_int)[2:]
                for i, used in enumerate(binary_code[::-1]):
                    if used == '1':
                        t = self.types[self.types.index(2**i)]
                        t_list.append(t)
                return t_list
            else:
                return [t]

    def generate_label(self, label_description):
        if isinstance(label_description, str):
            type_list = []
            weight_list = []
            label_description = label_description.replace(' ', '')
            label_description = label_description.replace(';', '+')
            label_description = label_description.replace(',', '+')
            types_for_label = label_description.split('+')

            for t in types_for_label:
                type_ident, _, weight = t.partition(':')
                type_list_addition = self.__get_type_from_str__(type_ident)
                n_additions = len(type_list_addition)
                type_list.extend(type_list_addition)
                if weight == '':
                    weight_list.extend(['*'] * n_additions)
                else:
                    weight_list.extend([float(weight)] * n_additions)
        elif isinstance(label_description, int):
            type_list = self.__get_type_from_str__(label_description)
            weight_list = ['*'] * len(type_list)
        return type_list, weight_list



