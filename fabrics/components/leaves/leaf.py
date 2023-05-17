from typing import List

import numpy as np
import casadi as ca

from fabrics.diffGeometry.energy import Lagrangian
from fabrics.diffGeometry.geometry import Geometry
from fabrics.helpers.variables import Variables
from fabrics.diffGeometry.diffMap import DifferentialMap

class Leaf(object):

    _lag: Lagrangian
    _geo: Geometry
    _map: DifferentialMap

    def __init__(
        self,
        parent_variables: Variables,
        leaf_name: str,
        forward_kinematics: ca.SX,
        dim: int = 1,
    ):
        self._parent_variables = parent_variables
        self._x = ca.SX.sym(f"x_{leaf_name}", dim)
        self._xdot = ca.SX.sym(f"xdot_{leaf_name}", dim)
        leaf_variables = Variables(
            state_variables={f"x_{leaf_name}": self._x, f"xdot_{leaf_name}": self._xdot}
        )
        self._leaf_variables = leaf_variables
        self._forward_kinematics = forward_kinematics
        self._p = {}
        self._leaf_name = leaf_name

    def set_params(self, **kwargs):
        for key in self._p:
            if key in kwargs:
                self._p[key] = kwargs.get(key)

    def geometry(self) -> Geometry:
        return self._geo

    def map(self) -> DifferentialMap:
        try:
            return self._map
        except AttributeError:
            return None

    def lagrangian(self):
        return self._lag

    def concretize(self) -> None:
        self._map.concretize()
        self._geo.concretize()
        self._lag.concretize()

    def evaluate(self, **kwargs) -> List[np.ndarray]:
        x, J, Jdot = self._map.forward(**kwargs)
        xdot = np.dot(J, kwargs['qdot'])
        state_variable_names = list(self._geo._vars.state_variables().keys())
        task_space_arguments = {
                state_variable_names[0]:x,
                state_variable_names[1]:xdot,
        }
        task_space_arguments.update(**kwargs)
        h, xddot = self._geo.evaluate(**task_space_arguments)
        M, f, H = self._lag.evaluate(**task_space_arguments)
        pulled_geo = self._geo.pull(self._map)
        pulled_geo.concretize()
        h_pulled, xddot_pulled = pulled_geo.evaluate(**kwargs)
        return dict(
            x= x,
            xdot=xdot,
            h=h,
            M=M,
            f=f,
            h_pulled=h_pulled
        )
