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

    def evaluate(self, **kwargs):
        for key in kwargs:
            assert isinstance(kwargs[key], np.ndarray)
        try:
            temp_dic = {}
            for i in kwargs:
                if i == "x_obst":
                    for j in range(len(kwargs[i])):
                       temp_dic[f"x_obst_{j}"] = kwargs[i][j]
                if i == "radius_obst":
                    for j in range(len(kwargs[i])):
                       temp_dic[f"radius_obst_{j}"] = kwargs[i][j]
                temp_dic[i]=kwargs[i]

            input_arrays = [temp_dic[i] for i in self._input_keys]
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




