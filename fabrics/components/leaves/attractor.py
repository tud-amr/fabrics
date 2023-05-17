import casadi as ca
import numpy as np

from fabrics.components.maps.parameterized_maps import ParameterizedGoalMap
from fabrics.diffGeometry.geometry import Geometry
from fabrics.diffGeometry.energy import Lagrangian
from fabrics.components.leaves.leaf import Leaf
from fabrics.helpers.variables import Variables
from fabrics.helpers.functions import parse_symbolic_input


class GenericAttractor(Leaf):
    """
    The GenericAttractor is a leaf to the tree of fabrics.

    The attractor's potential and metric are defined through the corresponding
    functions to which the symbolic expression is passed as a string.
    """

    def __init__(
        self, root_variables: Variables, fk_goal: ca.SX, attractor_name: str
    ):
        goal_dimension = fk_goal.size()[0]
        super().__init__(
            root_variables,
            f"{attractor_name}_leaf",
            fk_goal,
            dim=goal_dimension,
        )
        self.set_forward_map(attractor_name)

    def set_forward_map(self, goal_name):
        reference_name = f"x_{goal_name}"
        weight_name = f"weight_{goal_name}"
        goal_dimension = self._forward_kinematics.size()[0]
        if reference_name in self._parent_variables.parameters():
            reference_variable = self._parent_variables.parameters()[
                reference_name
            ]
        else:
            reference_variable = ca.SX.sym(reference_name, goal_dimension)
        if weight_name in self._parent_variables.parameters():
            weight_variable = self._parent_variables.parameters()[
                weight_name
            ]
        else:
            weight_variable = ca.SX.sym(weight_name, 1)
        self._geo_parameters = {
            reference_name: reference_variable,
            weight_name: weight_variable
        }
        self._weight = weight_variable
        self._leaf_variables.add_parameters(self._geo_parameters)
        self._parent_variables.add_parameters(self._geo_parameters)
        self._map = ParameterizedGoalMap(
            self._parent_variables, self._forward_kinematics, reference_variable
        )

    def set_potential(self, potential_expression: str) -> None:
        x = self._x
        xdot = self._xdot
        new_parameters, potential = parse_symbolic_input(potential_expression, x, xdot, name=self._leaf_name)
        psi = self._weight * potential
        self._parent_variables.add_parameters(new_parameters)
        h_psi = ca.gradient(psi, x)
        self._geo = Geometry(h=h_psi, var=self._leaf_variables)

    def set_metric(self, attractor_metric_expression: str) -> None:
        x = self._leaf_variables.position_variable()
        xdot = self._leaf_variables.velocity_variable()
        new_parameters, attractor_metric = parse_symbolic_input(attractor_metric_expression, x, xdot, name=self._leaf_name)
        self._parent_variables.add_parameters(new_parameters)
        lagrangian_psi = ca.dot(xdot, ca.mtimes(attractor_metric, xdot))
        self._lag = Lagrangian(lagrangian_psi, var=self._leaf_variables)

