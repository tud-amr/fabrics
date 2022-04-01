import casadi as ca
import numpy as np

from fabrics.planner.default_geometries import CollisionGeometry
from fabrics.planner.default_energies import CollisionLagrangian

from fabrics.diffGeometry.diffMap import DifferentialMap

from fabrics.leaves.leaf import Leaf

class ObstacleLeaf(Leaf):
    def __init__(self, q: ca.SX, qdot: ca.SX):
        super().__init__(q, qdot)
        self._p = {"lam": 2, 'exp': 1}

    def set_map(self, fk: ca.SX, position: np.ndarray, radius_obst: float, radius_body: float = 0.0):
        self._phi = ca.norm_2(fk - position) / (radius_obst + radius_body) - 1
        self._x = ca.SX.sym("x", 1)
        self._xdot = ca.SX.sym("xdot", 1)

    def concretize(self):
        self._dm = DifferentialMap(self._phi, q=self._q, qdot=self._qdot, Jdot_sign=-1)
        self._lag = CollisionLagrangian(self._x, self._xdot)
        self._geo = CollisionGeometry(self._x, self._xdot, **self._p)
