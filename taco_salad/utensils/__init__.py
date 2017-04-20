from __future__ import absolute_import, print_function, division
import pandas as pd


class LabelManager:
    def __init__(self):
        self.type_register = TypeDict()

    def __register_component__(self, component):
        if ' ' in component:
            raise AttributeError('Component names can not contain blanks!')
        if len(self.type_register) == 0:
            type_id = 0
        else:
            type_id = 2**(len(self.type_register) - 1)
        self.type_register[component] = type_id
        return type_id

    def generate_type(self, df, component):
        if isinstance(component, str):
            type_id = self.__register_component__(component)
            type_series = pd.Series(type_id,
                                    index=df.index,
                                    dtype=int)

        elif isinstance(component, dict):
            func_list = []
            component_ids = []
            default_idx = None
            for i, component_name in enumerate(component.keys()):
                comp_id = self.__register_component__(component_name)
                comp_func = component[component_name]
                if isinstance(comp_func, bool):
                    if comp_func:
                        if default_idx is None:
                            default_idx = i
                        else:
                            raise AttributeError('Second component set as'
                                                 'default!')
                    component_ids.append(comp_id)
                    func_list.append(comp_func)

            def decision_func(entry):
                for i, [comp_id, comp_func] in enumerate(zip(component_ids,
                                                             func_list)):
                    if i != default_idx:
                        if comp_func(df):
                            return comp_id
                return component_ids[default_idx]

            type_series = df.apply(decision_func)
        return type_series

    def __create_label_dict__(self, type_definition):
        if isinstance(type_definition, str):
            type_definition = type_definition.replace(' ', '')
            type_definition = type_definition.split('+')

        if isinstance(type_definition, list) or \
                isinstance(type_definition, tuple):
            type_definition = sum([self.type_register[comp_i]
                                   for comp_i in type_definition])

        if isinstance(type_definition, int):
            code = type_definition
        else:
            try:
                code = int(sum(type_definition))
            except:
                raise AttributeError('\'type_definition\' must be '
                                     ' list/tuple/int!')
        used_list = []
        binary_code = bin(code)
        assert len(binary_code) <= 8, 'Type code must be between 1 and 63'
        adjusted_binary_code = '%6s' % binary_code[2:]
        adjusted_binary_code = adjusted_binary_code.replace(' ', '0')
        label_dict = {}
        for exponent, used in enumerate(adjusted_binary_code[::-1]):
            component = 2 ** exponent
            if int(used) == 1:
                label_dict[component] = 1
                used_list.append(self.type_register[component])
            else:
                label_dict[component] = 0
        return label_dict

    def generate_label_dict(self, sig_definition, bkg_definition):
        sig_dict = self.__create_label_dict__(sig_definition)
        bkg_dict = self.__create_label_dict__(bkg_definition)
        label_dict = {}
        for i in range(len(self.type_register) - 1):
            component = 2**i
            component_name = self.type_register[component]
            is_sig = sig_dict[component] == 1
            is_bkg = bkg_dict[component] == 1
            if is_sig and is_bkg:
                raise('{} used as sig and bkg!'.format(component_name))
            elif is_sig:
                label_dict[component] = 1
            elif is_bkg:
                label_dict[component] = 0
            else:
                label_dict[component] = 2
        return label_dict

    def generate_label(self, df,
                       sig_definition=None,
                       bkg_definiton=None,
                       label_dict=None):
        if sig_definition is not None and bkg_definiton is not None:
            label_dict = self.generate_label(sig_definition=sig_definition,
                                             bkg_definiton=bkg_definiton)
        if not isinstance(label_dict, dict):
            raise AttributeError('Invalid \'label_dict\'!')

        def translate(x):
            return label_dict[x]
        return df['Type'].map(translate)

    def get_type_register(self):
        return self.type_register

    def convert_type(self, type_code):
        return self.type_register[type_code]


class TypeDict(dict):
    def __init__(self, one_way_dict):
        for key, value in one_way_dict.items():
            self.__setitem__(key, value)

    def __setitem__(self, key, value):
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
        return dict.__len__(self) // 2
