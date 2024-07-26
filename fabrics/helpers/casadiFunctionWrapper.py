import casadi as ca
from fabrics.helpers.variables import Variables
import numpy as np
import os
import pickle
import _pickle as cPickle
import bz2
import logging


class InputMissmatchError(Exception):
    pass


class CasadiFunctionWrapper(object):

    def __init__(self, name: str, variables: Variables, expressions: dict):
        self._name = name
        self._inputs = variables.asDict()
        self._expressions = expressions
        self._argument_dictionary = variables.parameters_values()
        self.create_function()

    def create_function(self):
        input_values = []
        input_keys = []
        expression_keys = []
        expression_values = []
        for input_key, input_value in self._inputs.items():
            input_keys.append(input_key)
            input_values.append(input_value)
        for expression_key, expression_value in self._expressions.items():
            expression_keys.append(expression_key)
            expression_values.append(expression_value)
        self._function = ca.Function(self._name, input_values, expression_values, input_keys, expression_keys)

    def function(self) -> ca.Function:
        return self._function

    def serialize(self, file_name):
        with bz2.BZ2File(file_name, 'w') as f:
            pickle.dump(self._function.serialize(), f)
            pickle.dump(self._argument_dictionary, f)

    def evaluate(self, **kwargs):
        self.process_inputs(**kwargs)
        try:
            output_dict = self._function(**self._argument_dictionary)
        except NotImplementedError:
            expected_inputs = list(self._inputs.keys())
            received_inputs = list(self._argument_dictionary.keys())
            unique_expected = [x for x in expected_inputs if x not in received_inputs]
            unique_received = [x for x in received_inputs if x not in expected_inputs]

            msg = "Inputs do not match\n"
            msg += f"Found unexpected inputs: {unique_received}\n"
            msg += f"Found missing inputs: {unique_expected}\n"
            raise InputMissmatchError(msg)
        for key, value in output_dict.items():
            if value.size() == (1, 1):
                output_dict[key] = np.array(value)[:, 0]
            elif value.size()[1] == 1:
                output_dict[key] = np.array(value)[:, 0]
            else:
                output_dict[key] = np.array(value)
        return output_dict

    def process_inputs(self, **kwargs):
        for key in kwargs: # pragma no cover
            if key == 'x_obst' or key == 'x_obsts':
                obstacle_dictionary = {}
                for j, x_obst_j in enumerate(kwargs[key]):
                    obstacle_dictionary[f'x_obst_{j}'] = x_obst_j
                self._argument_dictionary.update(obstacle_dictionary)
            elif key == 'radius_obst' or key == 'radius_obsts':
                radius_dictionary = {}
                for j, radius_obst_j in enumerate(kwargs[key]):
                    radius_dictionary[f'radius_obst_{j}'] = radius_obst_j
                self._argument_dictionary.update(radius_dictionary)
            elif key == 'x_obst_dynamic' or key == 'x_obsts_dynamic':
                obstacle_dyn_dictionary = {}
                for j, x_obst_dyn_j in enumerate(kwargs[key]):
                    obstacle_dyn_dictionary[f'x_obst_dynamic_{j}'] = x_obst_dyn_j
                self._argument_dictionary.update(obstacle_dyn_dictionary)
            elif key == 'xdot_obst_dynamic' or key == 'xdot_obsts_dynamic':
                xdot_dyn_dictionary = {}
                for j, xdot_obst_dyn_j in enumerate(kwargs[key]):
                    xdot_dyn_dictionary[f'xdot_obst_dynamic_{j}'] = xdot_obst_dyn_j
                self._argument_dictionary.update(xdot_dyn_dictionary)
            elif key == 'xddot_obst_dynamic' or key == 'xddot_obsts_dynamic':
                xddot_dyn_dictionary = {}
                for j, xddot_obst_dyn_j in enumerate(kwargs[key]):
                    xddot_dyn_dictionary[f'xddot_obst_dynamic_{j}'] = xddot_obst_dyn_j
                self._argument_dictionary.update(xddot_dyn_dictionary)
            elif key == 'radius_obst_dynamic' or key == 'radius_obsts_dynamic':
                radius_dyn_dictionary = {}
                for j, radius_obst_dyn_j in enumerate(kwargs[key]):
                    radius_dyn_dictionary[f'radius_obst_dynamic_{j}'] = radius_obst_dyn_j
                self._argument_dictionary.update(radius_dyn_dictionary)
            elif key == 'x_obst_cuboid' or key == 'x_obsts_cuboid':
                x_obst_cuboid_dictionary = {}
                for j, x_obst_cuboid_j in enumerate(kwargs[key]):
                    x_obst_cuboid_dictionary[f'x_obst_cuboid_{j}'] = x_obst_cuboid_j
                self._argument_dictionary.update(x_obst_cuboid_dictionary)
            elif key == 'size_obst_cuboid' or key == 'size_obsts_cuboid':
                size_obst_cuboid_dictionary = {}
                for j, size_obst_cuboid_j in enumerate(kwargs[key]):
                    size_obst_cuboid_dictionary[f'size_obst_cuboid_{j}'] = size_obst_cuboid_j
                self._argument_dictionary.update(size_obst_cuboid_dictionary)
            elif key.startswith('radius_body') and key.endswith('links'):
                # Radius bodies can be passed using a dictionary where the keys are simple integers.
                radius_body_dictionary = {}
                body_size_inputs = [input_exp for input_exp in list(self._inputs.keys()) if input_exp.startswith('radius_body')]
                for link_nr, radius_body_j in kwargs[key].items():
                    try:
                        key = [body_size_input for body_size_input in body_size_inputs if str(link_nr) in body_size_input][0]
                    except IndexError as e:
                        logging.warning(f"No body link with index {link_nr} in the inputs. Body link {link_nr} is ignored.")
                    radius_body_dictionary[key] = radius_body_j
                self._argument_dictionary.update(radius_body_dictionary)
            else:
                self._argument_dictionary[key] = kwargs[key]


class CasadiFunctionWrapper_deserialized(CasadiFunctionWrapper):

    def __init__(self, file_name: str):
        if os.path.isfile(file_name):
            logging.info(f"Initializing casadiFunctionWrapper from {file_name}")
            data = bz2.BZ2File(file_name, 'rb')
            self._function = ca.Function().deserialize(cPickle.load(data))
            self._argument_dictionary = cPickle.load(data)
            self._isload = True


