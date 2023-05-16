import casadi as ca
import numpy as np

from fabrics.components.maps.parameterized_maps import (
    ParameterizedObstacleMap,
)
from fabrics.diffGeometry.diffMap import DifferentialMap, ExplicitDifferentialMap
from fabrics.diffGeometry.geometry import Geometry
from fabrics.diffGeometry.energy import Lagrangian
from fabrics.components.leaves.leaf import Leaf
from fabrics.helpers.variables import Variables
from fabrics.helpers.functions import parse_symbolic_input


class GenericGeometryLeaf(Leaf):
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

class LimitLeaf(GenericGeometryLeaf):
    """
    The LimitLeaf is geometry leaf for joint limits.

    This leaf is not parameterized as the joint limits remain static for one robot.
    """
    def __init__(
        self,
        parent_variables: Variables,
        joint_index: int,
        limit: float,
        limit_index: int,
    ):
        limit_name = f"limit_joint_{joint_index}_{limit_index}"
        if limit_index == 0:
            phi = parent_variables.position_variable()[joint_index] - limit
        elif limit_index == 1:
            phi = limit - parent_variables.position_variable()[joint_index]
        else:
            print("There are only two limits.")
        super().__init__(
            parent_variables,
            f"{limit_name}_leaf",
            phi,
        )
        self.set_forward_map()

    def set_forward_map(self):
        self._map = DifferentialMap(self._forward_kinematics, self._parent_variables)

class SelfCollisionLeaf(GenericGeometryLeaf):
    """
    The SelfCollisionLeaf is a geometry leaf for self collision avoidanceself.

    This leaf is not parameterized as it is not changing at runtimeself.
    """
    def __init__(
        self,
        parent_variables: Variables,
        forward_kinematics: ca.SX,
        self_collision_name: str,
        self_collision_body_radius: float=0.10,
    ):
        super().__init__(
            parent_variables, self_collision_name, forward_kinematics
        )
        self.set_forward_map(self_collision_body_radius)

    def set_forward_map(self, self_collision_body_radius):
        phi = (
            ca.norm_2(self._forward_kinematics)
            / (2 * self_collision_body_radius) - 1
        )
        self._map = DifferentialMap(phi, self._parent_variables)


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
        collision_link: str,
    ):
        super().__init__(
            parent_variables, f"{obstacle_name}_{collision_link}_leaf", forward_kinematics
        )
        self.set_forward_map(obstacle_name, collision_link)

    def set_forward_map(self, obstacle_name, collision_link):
        radius_name = f"radius_{obstacle_name}"
        reference_name = f"x_{obstacle_name}"
        radius_body_name = f"radius_body_{collision_link}"
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
        self._map = ParameterizedObstacleMap(
            self._parent_variables,
            self._forward_kinematics,
            reference_variable,
            radius_variable,
            radius_body_variable,
        )



class ESDFGeometryLeaf(GenericGeometryLeaf):
    """ESDFGeometryLeaf is a leaf with explicit gradients that can be set
    at runtime.

    Euclidean Signed Distance Fields (ESDF) can be exploited to avoid explicit
    geometry representations. As automated differentiation does not work on ESDFs, 
    this GeometryLeaf adds parameters for J and Jdot that can be computed at
    runtime. Note that the signed distance to the closest obstacle is given
    by phi which is a function of a Euclidiean position which depends on the
    forward kinematics of the robot, phi(fk(q)). When computed the gradient, 
    one has to respect the chain rule leading to the jacobian of phi as:
    d phi / d q = d phi / d x * d x / d q.
    The second part is the gradient of the forward kinematics which can be 
    auto generated, see J_collision_link in set_forward_map.
    At runtime, one has to specify phi and d phi / d x.
    """
    def __init__(
            self,
            parent_variables: Variables,
            collision_link: str,
            collision_fk: ca.SX,
    ):
        self._collision_link = collision_link
        self._collision_fk = collision_fk
        phi = ca.SX.sym(f"esdf_phi_{self._collision_link}", 1)
        super().__init__(
            parent_variables,
            f"esdf_leaf_{collision_link}",
            phi,
        )
        self.set_forward_map()

    def set_forward_map(self):
        q = self._parent_variables.position_variable()
        J_collision_link = ca.jacobian(self._collision_fk, q)
        J_esdf = ca.transpose(ca.SX.sym(f"esdf_J_{self._collision_link}", 3))
        Jdot_esdf = ca.transpose(ca.SX.sym(f"esdf_Jdot_{self._collision_link}", q.size()[0]))
        J = ca.mtimes(J_esdf, J_collision_link)
        Jdot = Jdot_esdf
        radius_body_name = f"radius_body_{self._collision_link}"
        explicit_jacobians = {
            f"esdf_phi_{self._collision_link}": self._forward_kinematics,
            f"esdf_J_{self._collision_link}": J_esdf,
            f"esdf_Jdot_{self._collision_link}": Jdot_esdf,
        }
        self._parent_variables.add_parameters(explicit_jacobians)
        if radius_body_name in self._parent_variables.parameters():
            radius_body_variable = self._parent_variables.parameters()[
                radius_body_name
            ]
        else:
            radius_body_variable = ca.SX.sym(radius_body_name, 1)
        geo_parameters = {
            radius_body_name: radius_body_variable,
        }
        self._parent_variables.add_parameters(geo_parameters)
        phi_reduced = self._forward_kinematics - radius_body_variable
        self._map = ExplicitDifferentialMap(
            phi_reduced,
            self._parent_variables,
            J=J,
            Jdot=Jdot,
        )

