from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner
from fabrics.helpers.casadiFunctionWrapper import CasadiFunctionWrapper


import casadi as ca
import pickle
import numpy as np
from dataclasses import dataclass, field
from typing import Dict


class InputMissmatchError(Exception):
    pass

@dataclass
class FabricPlannerConfig:
    base_inertia: float = 0.2
    #s = -0.5 * (ca.sign(xdot) - 1)
    #h = -p["lam"] / (x ** p["exp"]) * s * xdot ** 2
    collision_geometry: str = (
        "-2 / (x ** 2) * (-0.5 * (ca.sign(xdot) - 1)) * xdot ** 2"
    )
    collision_finsler: str = (
        "2.0/(x**1) * xdot**2"
    )
    self_collision_geometry: str = (
        "-0.5 * / (x ** 2) * (-0.5 * (ca.sign(xdot) - 1) * xdot ** 2"
    )
    attractor_potential: str = (
        "5.0 * (ca.norm_2(x) + 1 / 10 * ca.log(1 + ca.exp(-2 * 10 * ca.norm_2(x))))"
    )
    attractor_metric: str = (
        "((2.0 - 0.3) * ca.exp(-1 * (0.75 * ca.norm_2(x))**2) + 0.3) * ca.SX(np.identity(x.size()[0]))"
    )
    ex_factor: float = 1.0
    damper: Dict[str, float] = field(
        default_factory=lambda: (
            {
                "alpha_b": 0.5,
                "alpha_eta": 0.5,
                "alpha_shift": 0.5,
                "beta_distant": 0.01,
                "beta_close": 6.5,
                "radius_shift": 0.02,
            }
        )
    )


class SerializedFabricPlanner(ParameterizedFabricPlanner):
    def __init__(self, dof: int, robot_type: str, **kwargs):
        ParameterizedFabricPlanner.__init__(self, dof, robot_type, **kwargs)
        # super(ParameterizedFabricPlanner, self).__init__()
        self._expressions = {"xddot": 0}

    def concretize_serialized(self, file_name: str):
        try:
            a_ex = (
                self._eta * self._geometry._alpha
                + (1 - self._eta) * self._forced_geometry._alpha
            )
            beta_subst = self._beta.substitute(-a_ex, -self._geometry._alpha)
            xddot = self._forced_geometry._xddot - (a_ex + beta_subst) * self._geometry.xdot()
        except AttributeError:
            print("No forcing term, using pure geoemtry")
            self._geometry.concretize()
            xddot = self._geometry._xddot - self._geometry._alpha * self._geometry._vars.velocity_variable()
        self._funs = CasadiFunctionWrapper(
            "funs", self.variables.asDict(), {"xddot": xddot}
        )
        with open(file_name, 'wb') as f:
            pickle.dump(self._funs._function.serialize(), f)
            pickle.dump(self._funs._input_keys, f)


    """ RUNTIME METHODS """

    def evaluate(self, function, input_keys, **kwargs):
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
            input_arrays = [argument_dictionary[i] for i in input_keys]
        except KeyError as e:
            msg = f"Key {e} is not contained in the inputs\n"
            msg += f"Possible keys are {input_keys}\n"
            msg += f"You prorvided {list(kwargs.keys())}\n"
            raise InputMissmatchError(msg)
        list_array_outputs = function(*input_arrays)
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

    def serialized_compute_action(self, function, input_keys, **kwargs):
            """
            Computes action based on the states passed.

            The variables passed are the joint states, and the goal position.
            """
            evaluations = self.evaluate(function, input_keys, **kwargs)
            action = evaluations["xddot"]
            """
            # avoid to small actions
            if np.linalg.norm(action) < eps:
                action = np.zeros(self._n)
            """
            return action

