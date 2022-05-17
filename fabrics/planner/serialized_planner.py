from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner
from fabrics.helpers.casadiFunctionWrapper import CasadiFunctionWrapper


import casadi as ca
import pickle
import numpy as np
import os
from typing import Dict


class InputMissmatchError(Exception):
    pass


class SerializedFabricPlanner(ParameterizedFabricPlanner):
    def __init__(self, dof: int, robot_type: str, file_name: str, **kwargs):
        ParameterizedFabricPlanner.__init__(self, dof, robot_type, **kwargs)
        self._isload = False
        if os.path.isfile(file_name):
            print(f"Initializing planner from {file_name}")
            with open(file_name, 'rb') as f:
                self._funs = ca.Function().deserialize(pickle.load(f))
                self._input_keys = pickle.load(f)
            self._isload = True
        # this is only used in evaluate() and only the key is used there
        self._expressions = {"xddot": 0}

    """ RUNTIME METHODS """

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
        list_array_outputs = self._funs(*input_arrays)
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

    def serialized_compute_action(self, **kwargs):
            """
            Computes action based on the states passed.

            The variables passed are the joint states, and the goal position.
            """
            evaluations = self.evaluate(**kwargs)

            action = evaluations["xddot"]
            """
            # avoid to small actions
            if np.linalg.norm(action) < eps:
                action = np.zeros(self._n)
            """
            return action

