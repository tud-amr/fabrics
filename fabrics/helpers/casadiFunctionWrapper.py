import casadi as ca
import numpy as np

class CasadiFunctionWrapper(object):

    def __init__(self, name: str, inputs: dict, expressions: dict):
        self._name = name
        self._inputs = inputs
        self._expressions = expressions
        self.create_function()

    def create_function(self):
        list_inputs = [self._inputs[i] for i in sorted(self._inputs.keys())]
        self._list_expressions = [self._expressions[i] for i in sorted(self._expressions.keys())]
        self._function = ca.Function(self._name, list_inputs, self._list_expressions)

    def evaluate(self, inputs: dict):
        list_array_inputs = [inputs[i] for i in sorted(inputs.keys())]
        list_array_outputs = self._function(*list_array_inputs)
        output_dict = {}
        for i, key in enumerate(sorted(self._expressions.keys())):
            output_dict[key] = list_array_outputs[i]
        return output_dict




