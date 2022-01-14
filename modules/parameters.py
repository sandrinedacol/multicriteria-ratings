from collections import *


def convert_string_to_parameter(value):
    if value in ['None', 'none', 'NONE', 'NA', 'nan', 'NaN']:
        parameter = None
    else:
        parameter = value
    return parameter

class Parameters():

    def __init__(self, yml_dict):
        self.params_names = ['mode', 'dataset_name', 'n_reviews', 'criteria', 'metric', 'sentiment_analyzer', 'theta', 'alpha', 'stars_averaging']
        if yml_dict['mode'] == 'test':
            self.params_names += ['tested_parameter', 'min_value', 'max_value', 'n_values', 'significant_digits', 'loss_type']
        else:
            self.params_names += ['error_max']

        self.params = dict()
        for param_name in self.params_names:
            setattr(self, param_name, convert_string_to_parameter(yml_dict[param_name]))
