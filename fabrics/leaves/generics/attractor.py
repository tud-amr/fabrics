import casadi as ca
import numpy as np

from fabrics.planner.default_maps import ParameterizedGoalMap
from fabrics.diffGeometry.geometry import Geometry
from fabrics.diffGeometry.energy import Lagrangian
from fabrics.leaves.leaf import Leaf
from fabrics.helpers.variables import Variables

class GenericAttractor(Leaf):
    def __init__(self, root_variables: Variables, fk_goal: ca.SX):
        super().__init__(root_variables)
        self._fk = fk_goal
        x = ca.SX.sym("x_goal", fk_goal.size()[0])
        xdot = ca.SX.sym("xdot_goal", fk_goal.size()[0])
        self._var_psi = Variables(state_variables={"x": x, "xdot": xdot})
        self._forward_map = ParameterizedGoalMap(self._var_q, self._fk)

    def set_potential(self, potential: str) -> None:
        x_goal = self._var_psi.position_variable()
        psi = eval(potential)
        h_psi = ca.gradient(psi, x_goal)
        self._geo = Geometry(h=h_psi, var=self._var_psi)

    def set_metric(self, attractor_metric: str) -> None:
        x_goal = self._var_psi.position_variable()
        xdot_goal = self._var_psi.velocity_variable()
        M_psi = eval(attractor_metric)
        lagrangian_psi = ca.dot(xdot_goal, ca.mtimes(M_psi, xdot_goal))
        self._lag = Lagrangian(lagrangian_psi, var=self._var_psi)

    def map(self):
        return self._forward_map
