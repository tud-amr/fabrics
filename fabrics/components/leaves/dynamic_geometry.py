import casadi as ca
import numpy as np

from fabrics.diffGeometry.diffMap import DifferentialMap
from fabrics.components.maps.parameterized_maps import (
    ParameterizedObstacleMap,
)
from fabrics.diffGeometry.geometry import Geometry
from fabrics.diffGeometry.energy import Lagrangian
from fabrics.components.leaves.dynamic_leaf import DynamicLeaf
from fabrics.helpers.variables import Variables
from fabrics.helpers.functions import parse_symbolic_input


class GenericDynamicGeometryLeaf(DynamicLeaf):
    """
    The GenericDyanmicGeometry is a leaf to the tree of fabrics.

    The geometry's geometry and metric are defined through the corresponding
    functions to which the symbolic expression is passed as a string.
    In contrast to the GenericGeometry, the GenericDynamicGeometry has an
    additional differential map, namely a RelativeDifferentiaMap.
    """

    def set_geometry(self, geometry: str) -> None:
        """
        Sets the geometry from a string.

        Params
        ---------
        geometry: str
            String that holds the geometry. The variables must be x, xdot.
        """
        x = self._x
        xdot = self._xdot
        new_parameters, h_geometry = parse_symbolic_input(geometry, x, xdot, name=self._leaf_name)
        self._parent_variables.add_parameters(new_parameters)
        self._geo = Geometry(h=h_geometry, var=self._leaf_variables)

    def set_finsler_structure(self, finsler_structure: str) -> None:
        """
        Sets the Finsler structure from a string.

        Params
        ---------
        finsler_structure: str
            String that holds the Finsler structure. The variables must be x, xdot.
        """
        x = self._x
        xdot = self._xdot
        new_parameters, lagrangian_geometry = parse_symbolic_input(finsler_structure, x, xdot, name=self._leaf_name)
        self._parent_variables.add_parameters(new_parameters)
        self._lag = Lagrangian(lagrangian_geometry, var=self._leaf_variables)


class DynamicObstacleLeaf(GenericDynamicGeometryLeaf):
    """
    The DynamicObstacleLeaf is a geometry leaf for spherical obstacles.

    The obstacles are parameterized by the obstacles radius, its position,
    velocity, acceleration and the radius of the englobing sphere for the
    corresponding link. Moreover, the symbolic expression for the forward
    expression is passed to the constructor.
    """

    def __init__(
        self,
        parent_variables: Variables,
        forward_kinematics: ca.SX,
        obstacle_name: str,
        collision_link: str,
        reference_parameters: dict = None,
    ):
        dim_ref = forward_kinematics.size()[0]
        super().__init__(
            parent_variables,
            f"{obstacle_name}_{collision_link}_leaf",
            forward_kinematics,
            dim = 1,
            dim_ref = dim_ref,
            reference_parameters=reference_parameters,
        )
        self.set_forward_map(obstacle_name, collision_link)

    def set_forward_map(self, obstacle_name, collision_link):
        radius_name = f"radius_{obstacle_name}"
        radius_body_name = f"radius_body_{collision_link}"
        obstacle_dimension = self._forward_kinematics.size()[0]
        if radius_name in self._parent_variables.parameters():
            radius_variable = self._parent_variables.parameters()[radius_name]
        else:
            radius_variable = ca.SX.sym(radius_name, 1)
        if radius_body_name in self._parent_variables.parameters():
            radius_body_variable = self._parent_variables.parameters()[
                radius_body_name
            ]
        else:
            radius_body_variable = ca.SX.sym(radius_body_name, 1)
        geo_parameters = {
            radius_name: radius_variable,
            radius_body_name: radius_body_variable,
        }
        self._parent_variables.add_parameters(geo_parameters)
        self._forward_map = DifferentialMap(self._forward_kinematics, self._parent_variables)
        self._geometry_map = ParameterizedObstacleMap(
            self._relative_variables,
            self._relative_variables.position_variable(),
            np.zeros(self._dim_ref),
            radius_variable,
            radius_body_variable,
        )

    def geometry_map(self):
        return self._geometry_map

    def map(self):
        return self._forward_map
