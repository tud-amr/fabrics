import casadi as ca
import numpy as np

from fabrics.diffGeometry.diffMap import DifferentialMap
from fabrics.diffGeometry.geometry import Geometry
from fabrics.diffGeometry.energy import Lagrangian
from fabrics.components.leaves.dynamic_leaf import DynamicLeaf
from fabrics.helpers.variables import Variables


class GenericDynamicAttractor(DynamicLeaf):
    """
    The GenericDyanmicAttractor is a leaf to the tree of fabrics.

    The attractor's potential and metric are defined through the corresponding
    functions to which the symbolic expression is passed as a string.
    In contrast to the GenericAttractor, the GenericDynamicAttractor has an
    additional differential map, namely a RelativeDifferentiaMap.
    """

    def __init__(
        self, root_variables: Variables,
        fk_goal: ca.SX,
        attractor_name: str,
    ):
        goal_dimension = fk_goal.size()[0]
        super().__init__(
            root_variables,
            f"{attractor_name}_leaf",
            fk_goal,
            dim_ref=goal_dimension,
            dim=goal_dimension
        )
        self.set_forward_map(attractor_name)

    def set_forward_map(self, goal_name):
        weight_name = f"weight_{goal_name}"
        if weight_name in self._parent_variables.parameters():
            weight_variable = self._parent_variables.parameters()[
                weight_name
            ]
        else:
            weight_variable = ca.SX.sym(weight_name, 1)
        geo_parameters = {
            weight_name: weight_variable
        }
        self._weight = weight_variable
        self._parent_variables.add_parameters(geo_parameters)
        self._forward_map = DifferentialMap(self._forward_kinematics, self._parent_variables)

    def set_potential(self, potential: str) -> None:
        x = self._x_rel
        xdot = self._xdot_rel
        psi = self._weight * eval(potential)
        h_psi = ca.gradient(psi, x)
        self._geo = Geometry(h=h_psi, var=self._relative_variables)

    def set_metric(self, attractor_metric: str) -> None:
        x = self._x_rel
        xdot = self._xdot_rel
        #x_ref = self._x_ref
        #xdot_ref = self._xdot_ref
        #xddot_ref = self._xddot_ref
        attractor_metric = eval(attractor_metric)
        lagrangian_psi = ca.dot(xdot, ca.mtimes(attractor_metric, xdot))
        self._lag = Lagrangian(lagrangian_psi, var=self._relative_variables)

    def map(self):
        return self._forward_map
