import casadi as ca
import numpy as np

from fabrics.planner.default_maps import (
    ParameterizedGeometryMap,
    ParameterizedObstacleMap,
)
from fabrics.diffGeometry.geometry import Geometry
from fabrics.diffGeometry.energy import Lagrangian
from fabrics.components.leaves.leaf import Leaf
from fabrics.helpers.variables import Variables


class GenericGeometryLeaf(Leaf):
    def set_geometry(self, geometry: str) -> None:
        """
        Sets the geometry from a string.

        Params
        ---------
        geometry: str
            String that holds the geometry. The variables must be x, xdot.
        """
        x = self._leaf_variables.position_variable()
        xdot = self._leaf_variables.velocity_variable()
        h_geometry = eval(geometry)
        self._geo = Geometry(h=h_geometry, var=self._leaf_variables)

    def set_finsler_structure(self, finsler_structure: str) -> None:
        """
        Sets the Finsler structure from a string.

        Params
        ---------
        finsler_structure: str
            String that holds the Finsler structure. The variables must be x, xdot.
        """
        x = self._leaf_variables.position_variable()
        xdot = self._leaf_variables.velocity_variable()
        lagrangian_geometry = eval(finsler_structure)
        self._lag = Lagrangian(lagrangian_geometry, var=self._leaf_variables)


class ObstacleLeaf(GenericGeometryLeaf):
    """
    The ObstacleLeaf is a geometry leaf for spherical obstacles.

    The obstacles are parameterized by the obstacles radius, its position and
    the radius of the englobing sphere for the corresponding link.
    Moreover, the symbolic expression for the forward expression is passed to
    the constructor.
    """

    def __init__(
        self,
        parent_variables: Variables,
        forward_kinematics: ca.SX,
        obstacle_name: str,
    ):
        super().__init__(
            parent_variables, f"{obstacle_name}_leaf", forward_kinematics
        )
        self.set_forward_map(obstacle_name)

    def set_forward_map(self, obstacle_name):
        radius_name = f"radius_{obstacle_name}"
        reference_name = f"x_{obstacle_name}"
        radius_body_name = f"radius_body"
        obstacle_dimension = self._forward_kinematics.size()[0]
        if radius_name in self._parent_variables.parameters():
            radius_variable = self._parent_variables.parameters()[radius_name]
        else:
            radius_variable = ca.SX.sym(radius_name, 1)
        if reference_name in self._parent_variables.parameters():
            reference_variable = self._parent_variables.parameters()[
                reference_name
            ]
        else:
            reference_variable = ca.SX.sym(reference_name, obstacle_dimension)
        if radius_body_name in self._parent_variables.parameters():
            radius_body_variable = self._parent_variables.parameters()[
                radius_body_name
            ]
        else:
            radius_body_variable = ca.SX.sym(radius_body_name, 1)
        geo_parameters = {
            reference_name: reference_variable,
            radius_name: radius_variable,
            radius_body_name: radius_body_variable,
        }
        self._parent_variables.add_parameters(geo_parameters)
        self._forward_map = ParameterizedObstacleMap(
            self._parent_variables,
            self._forward_kinematics,
            reference_variable,
            radius_variable,
            radius_body_variable,
        )

    def map(self):
        return self._forward_map
