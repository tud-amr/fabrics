import casadi as ca
import numpy as np

from fabrics.planner.default_geometries import GoalGeometry
from fabrics.planner.default_energies import GoalLagrangian
from fabrics.planner.default_maps import GoalMap, ParameterizedGoalMap

from fabrics.leaves.leaf import Leaf

class Attractor(Leaf):
    def __init__(self, q: ca.SX, qdot: ca.SX):
        super().__init__(q, qdot)
        self._p = {"k_psi": 10}

    def set_goal(self, goal: np.ndarray, fk: ca.SX):
        self._x = ca.SX.sym("x_psi", fk.size()[0])
        self._xdot = ca.SX.sym("xdot_psi", fk.size()[0])
        self._fk = fk
        self._goal = goal

    def x_psi(self):
        return self._x

    def concretize(self):
        self._dm = GoalMap(self._q, self._qdot, self._fk, self._goal)
        self._lag = GoalLagrangian(self._x, self._xdot)
        self._geo = GoalGeometry(self._x, self._xdot, k_psi=self._p['k_psi'])

class ParameterizedAttractor(Attractor):
    def __init__(self, q: ca.SX, qdot: ca.SX):
        super().__init__(q, qdot)

    def set_goal(self, goal: ca.SX, fk: ca.SX):
        self._x = ca.SX.sym("x_psi", fk.size()[0])
        self._xdot = ca.SX.sym("xdot_psi", fk.size()[0])
        self._fk = fk
        self._goal = goal

    def concretize(self):
        self._dm = ParameterizedGoalMap(self._q, self._qdot, self._fk, self._goal)
        self._lag = GoalLagrangian(self._x, self._xdot)
        self._geo = GoalGeometry(self._x, self._xdot, k_psi=self._p['k_psi'])
