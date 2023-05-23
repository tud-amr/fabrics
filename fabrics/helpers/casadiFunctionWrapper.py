import pdb
import casadi as ca
import numpy as np
import os
import pickle
import _pickle as cPickle
import bz2
import logging


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
        self._input_sizes = {i: self._inputs[i].size() for i in self._inputs}
        self._list_expressions = [self._expressions[i] for i in sorted(self._expressions.keys())]
        input_expressions = [self._inputs[i] for i in self._input_keys]
        self._function = ca.Function(self._name, input_expressions, self._list_expressions)

    def serialize(self, file_name):
        with bz2.BZ2File(file_name, 'w') as f:
            pickle.dump(self._function.serialize(), f)
            pickle.dump(list(self._expressions.keys()), f)
            pickle.dump(self._input_keys, f)

    def evaluate(self, **kwargs):
        argument_dictionary = {}
        for key in kwargs:
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
            if key == 'x_obst_dynamic' or key == 'x_obsts_dynamic':
                obstacle_dyn_dictionary = {}
                for j, x_obst_dyn_j in enumerate(kwargs[key]):
                    obstacle_dyn_dictionary[f'x_obst_dynamic_{j}'] = x_obst_dyn_j
                argument_dictionary.update(obstacle_dyn_dictionary)
            if key == 'xdot_obst_dynamic' or key == 'xdot_obsts_dynamic':
                xdot_dyn_dictionary = {}
                for j, xdot_obst_dyn_j in enumerate(kwargs[key]):
                    xdot_dyn_dictionary[f'xdot_obst_dynamic_{j}'] = xdot_obst_dyn_j
                argument_dictionary.update(xdot_dyn_dictionary)
            if key == 'xddot_obst_dynamic' or key == 'xddot_obsts_dynamic':
                xddot_dyn_dictionary = {}
                for j, xddot_obst_dyn_j in enumerate(kwargs[key]):
                    xddot_dyn_dictionary[f'xddot_obst_dynamic_{j}'] = xddot_obst_dyn_j
                argument_dictionary.update(xddot_dyn_dictionary)
            if key == 'radius_obst_dynamic' or key == 'radius_obsts_dynamic':
                radius_dyn_dictionary = {}
                for j, radius_obst_dyn_j in enumerate(kwargs[key]):
                    radius_dyn_dictionary[f'radius_obst_dynamic_{j}'] = radius_obst_dyn_j
                argument_dictionary.update(radius_dyn_dictionary)
            if key.startswith('radius_body') and key.endswith('links'):
                # Radius bodies can be passed using a dictionary where the keys are simple integers.
                radius_body_dictionary = {}
                body_size_inputs = [input_exp for input_exp in self._input_keys if input_exp.startswith('radius_body')]
                for link_nr, radius_body_j in kwargs[key].items():
                    try:
                        key = [body_size_input for body_size_input in body_size_inputs if str(link_nr) in body_size_input][0]
                    except IndexError as e:
                        logging.warning(f"No body link with index {link_nr} in the inputs. Body link {link_nr} is ignored.")
                    radius_body_dictionary[key] = radius_body_j

                argument_dictionary.update(radius_body_dictionary)
            else:
                argument_dictionary[key] = kwargs[key]
        input_arrays = []
        try:
            for i in self._input_keys:
                """
                if not argument_dictionary[i].size == self._input_sizes[i][0] * self._input_sizes[i][1]:
                    raise InputMissmatchError(f"Size of input argument {i} with size {argument_dictionary[i].size} does not match size required {self._input_sizes[i][0]}")
                """
                input_arrays.append(argument_dictionary[i])
            input_arrays = [argument_dictionary[i] for i in self._input_keys]
        except KeyError as e:
            msg = f"Key {e} is not contained in the inputs\n"
            msg += f"Possible keys are {self._input_keys}\n"
            msg += f"You provided {list(kwargs.keys())}\n"
            raise InputMissmatchError(msg)
        try:
            list_array_outputs = self._function(*input_arrays)
        except RuntimeError as runtime_error:
            raise InputMissmatchError(runtime_error.args)
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


class CasadiFunctionWrapper_deserialized(CasadiFunctionWrapper):

    def __init__(self, file_name: str):
        if os.path.isfile(file_name):
            logging.info(f"Initializing casadiFunctionWrapper from {file_name}")
            data = bz2.BZ2File(file_name, 'rb')
            self._function = ca.Function().deserialize(cPickle.load(data))
            expression_keys = cPickle.load(data)
            self._input_keys = cPickle.load(data)
            self._expressions = {}
            for key in expression_keys:
                self._expressions[key] = []
            self._isload = True


