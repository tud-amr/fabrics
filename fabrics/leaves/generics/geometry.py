import casadi as ca
import numpy as np

from fabrics.planner.default_maps import ParameterizedGeometryMap, ParameterizedObstacleMap
from fabrics.diffGeometry.geometry import Geometry
from fabrics.diffGeometry.energy import Lagrangian
from fabrics.leaves.leaf import Leaf
from fabrics.helpers.variables import Variables

class GenericGeometry(Leaf):
    def __init__(self, root_variables: Variables, fk: ca.SX):
        super().__init__(root_variables)
        self._fk = fk
        x = ca.SX.sym("x_geometry", 1)
        xdot = ca.SX.sym("xdot_geometry", 1)
        self._var_geometry = Variables(
                state_variables={"x": x, "xdot": xdot},
        )

    def set_geometry(self, geometry: str) -> None:
        x_geometry = self._var_geometry.position_variable()
        xdot_geometry = self._var_geometry.velocity_variable()
        h_geometry = eval(geometry)
        self._geo = Geometry(h=h_geometry, var=self._var_geometry)

    def set_finsler_structure(self, finsler_structe: str) -> None:
        x_geometry = self._var_geometry.position_variable()
        xdot_geometry = self._var_geometry.velocity_variable()
        lagrangian_geometry = eval(finsler_structe)
        self._lag = Lagrangian(lagrangian_geometry, var=self._var_geometry)


class ObstacleGeometry(GenericGeometry):
    def __init__(self, root_variables: Variables, fk: ca.SX, obstacle_name: str):
        super().__init__(root_variables, fk)
        radius_name = f"radius_{obstacle_name}"
        reference_name = f"x_{obstacle_name}"
        radius_body_name = f"radius_body"
        if radius_name in self._var_q.parameters():
            radius_variable = self._var_q.parameters()[radius_name]
        else:
            radius_variable = ca.SX.sym(radius_name, 1)
        if reference_name in self._var_q.parameters():
            reference_variable = self._var_q.parameters()[reference_name]
        else:
            reference_variable = ca.SX.sym(reference_name, fk.size()[0])
        if radius_body_name in self._var_q.parameters():
            radius_body_variable = self._var_q.parameters()[radius_body_name]
        else:
            radius_body_variable = ca.SX.sym(radius_body_name, 1)
        geo_parameters = {
                reference_name: reference_variable,
                radius_name: radius_variable, 
                radius_body_name: radius_body_variable
        }
        self._var_q.add_parameters(geo_parameters)
        self._forward_map = ParameterizedObstacleMap(
                self._var_q, self._fk, reference_variable, radius_variable, radius_body_variable
        )

    def map(self):
        return self._forward_map
