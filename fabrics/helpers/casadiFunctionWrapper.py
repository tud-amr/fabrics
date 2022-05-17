import casadi as ca
import numpy as np
from collections import OrderedDict

class InputMissmatchError(Exception):
    pass


class CasadiFunctionWrapper(object):

    def __init__(self, name: str, inputs: dict, expressions: dict):
        self._name = name
        self._inputs = inputs
        self._expressions = expressions
        self.create_function()

    def create_function(self):
        self._input_keys = sorted(tuple(self._inputs.keys()))
        self._list_expressions = [self._expressions[i] for i in sorted(self._expressions.keys())]
        input_expressions = [self._inputs[i] for i in self._input_keys]
        self._function = ca.Function(self._name, input_expressions, self._list_expressions)

    def serialize(self):
        with open(file_name, 'wb') as f:
            pickle.dump(self._function.serialize(), f)
            pickle.dump(self._input_keys, f)

    def evaluate(self, **kwargs):
        argument_dictionary = {}
        for key in kwargs:
            assert isinstance(kwargs[key], np.ndarray) or isinstance(kwargs[key], list)
            if key == 'x_obst' or key == 'x_obsts':
                obstacle_dictionary = {}
                for j, x_obst_j in enumerate(kwargs[key]):
                    obstacle_dictionary[f'x_obst_{j}'] = x_obst_j
                argument_dictionary.update(obstacle_dictionary)
            if key == 'radius_obst' or key == 'radius_obsts':
                radius_dictionary = {}
                for j, radius_obst_j in enumerate(kwargs[key]):
                    radius_dictionary[f'radius_obst_{j}'] = radius_obst_j
                argument_dictionary.update(radius_dictionary)
            else:
                argument_dictionary[key] = kwargs[key]
        try:
            input_arrays = [argument_dictionary[i] for i in self._input_keys]
        except KeyError as e:
            msg = f"Key {e} is not contained in the inputs\n"
            msg += f"Possible keys are {self._input_keys}\n"
            msg += f"You prorvided {list(kwargs.keys())}\n"
            raise InputMissmatchError(msg)
        list_array_outputs = self._function(*input_arrays)
        output_dict = {}
        if isinstance(list_array_outputs, ca.DM):
            return {list(self._expressions.keys())[0]: np.array(list_array_outputs)[:, 0]}
        for i, key in enumerate(sorted(self._expressions.keys())):
            raw_output = list_array_outputs[i]
            if raw_output.size() == (1, 1):
                output_dict[key] = np.array(raw_output)[:, 0]
            elif raw_output.size()[1] == 1:
                output_dict[key] = np.array(raw_output)[:, 0]
            else:
                output_dict[key] = np.array(raw_output)
        return output_dict




