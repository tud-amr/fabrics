import casadi as ca
from copy import deepcopy


class Variables(object):
    def __init__(self, state_variables={}, parameters={}):
        self._state_variables = state_variables
        self._parameters = parameters
        if len(state_variables) > 0:
            self._state_variable_names = list(state_variables.keys())
        if len(parameters) > 0:
            self._parameter_names = list(parameters.keys())

    def state_variables(self):
        return self._state_variables

    def add_state_variable(self, name, value):
        self._state_variables[name] = value

    def parameters(self):
        return self._parameters

    def add_parameter(self, name: str, value: ca.SX) -> None:
        self._parameters[name] = value

    def set_parameters(self, parameters):
        self._parameters = parameters

    def variable_by_name(self, name: str) -> ca.SX:
        return self._state_variables[name]

    def parameter_by_name(self, name: str) -> ca.SX:
        return self._parameters[name]

    def position_variable(self) -> ca.SX:
        return self.variable_by_name(self._state_variable_names[0])

    def velocity_variable(self) -> ca.SX:
        return self.variable_by_name(self._state_variable_names[1])

    def verify(self):
        for key in self._state_variables:
            assert isinstance(key, str)
            assert isinstance(self._state_variables[key], ca.SX)
        for key in self._parameters:
            assert isinstance(key, str)
            assert isinstance(self._parameters[key], ca.SX)

    def asDict(self):
        joinedDict = {}
        joinedDict.update(self._state_variables)
        joinedDict.update(self._parameters)
        return joinedDict

    def __add__(self, b):
        joined_state_variables = deepcopy(self._state_variables)
        for key, value in b.state_variables().items():
            if key in joined_state_variables:
                if ca.is_equal(joined_state_variables[key], value):
                    continue
                else:
                    new_key = key
                    counter = 1
                    while new_key in joined_state_variables.keys():
                        new_key = key + "_" + str(counter)
                        counter += 1
                    joined_state_variables[new_key] = value
            else:
                joined_state_variables[key] = value
        joined_parameters = deepcopy(self._parameters)
        for key, value in b.parameters().items():
            if key in joined_parameters:
                if ca.is_equal(joined_parameters[key], value):
                    continue
                else:
                    new_key = key
                    counter = 1
                    while new_key in joined_parameters.keys():
                        new_key = key + "_" + str(counter)
                        counter += 1
                    joined_parameters[new_key] = value
            else:
                joined_parameters[key] = value

        return Variables(state_variables=joined_state_variables, parameters=joined_parameters)

    def len(self):
        return len(self._parameters.values()) + len(
            self._state_variables.values()
        )

    def __str__(self):
        return (
            "State variables: "
            + self._state_variables.__str__()
            + "| parameters : "
            + self._parameters.__str__()
        )
