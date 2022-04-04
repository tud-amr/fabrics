import casadi as ca
import numpy as np

from fabrics.planner.default_geometries import CollisionGeometry
from fabrics.planner.default_energies import CollisionLagrangian

from fabrics.diffGeometry.diffMap import DifferentialMap

from fabrics.leaves.leaf import Leaf

from fabrics.helpers.variables import Variables


class ObstacleLeaf(Leaf):
    def __init__(self, var: Variables):
        super().__init__(var)
        self._p = {"lam": 2, "exp": 1}

    def set_map(
        self,
        fk: ca.SX,
        position: np.ndarray,
        radius_obst: float,
        radius_body: float = 0.0,
    ):
        self._phi = ca.norm_2(fk - position) / (radius_obst + radius_body) - 1
        x = ca.SX.sym("x", 1)
        xdot = ca.SX.sym("xdot", 1)
        self._var_x = Variables(state_variables={"x": x, "xdot": xdot})

    def concretize(self):
        self._dm = DifferentialMap(self._phi, var=self._var_q, Jdot_sign=-1)
        self._lag = CollisionLagrangian(self._var_x)
        self._geo = CollisionGeometry(self._var_x, **self._p)
