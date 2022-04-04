import casadi as ca
import numpy as np

from fabrics.planner.default_geometries import GoalGeometry
from fabrics.planner.default_energies import GoalLagrangian
from fabrics.planner.default_maps import GoalMap, ParameterizedGoalMap

from fabrics.leaves.leaf import Leaf

from fabrics.helpers.variables import Variables

class Attractor(Leaf):
    def __init__(self, var: Variables):
        super().__init__(var)
        self._p = {"k_psi": 10}

    def set_goal(self, goal: np.ndarray, fk: ca.SX):
        x = ca.SX.sym("x_psi", fk.size()[0])
        xdot = ca.SX.sym("xdot_psi", fk.size()[0])
        self._var_x = Variables(state_variables={'x': x, 'xdot': xdot})
        self._fk = fk
        self._goal = goal

    def x_psi(self):
        return self._var_x.position_variable()

    def concretize(self):
        self._dm = GoalMap(self._var_q, self._fk, self._goal)
        self._lag = GoalLagrangian(self._var_x)
        self._geo = GoalGeometry(self._var_x, k_psi=self._p['k_psi'])

class ParameterizedAttractor(Attractor):
    def __init__(self, var: Variables):
        super().__init__(var)

    def set_goal(self, x_goal: ca.SX, fk: ca.SX):
        x = ca.SX.sym("x_psi", fk.size()[0])
        xdot = ca.SX.sym("xdot_psi", fk.size()[0])
        self._var_x = Variables(state_variables={'x': x, 'xdot': xdot})
        self._var_q.set_parameters({'x_goal': x_goal})
        self._fk = fk
        self._goal = x_goal

    def concretize(self):
        self._dm = ParameterizedGoalMap(self._var_q, self._fk)
        self._lag = GoalLagrangian(self._var_x)
        self._geo = GoalGeometry(self._var_x, k_psi=self._p['k_psi'])
