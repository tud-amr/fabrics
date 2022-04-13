import casadi as ca
import numpy as np

from fabrics.planner.default_maps import ParameterizedGeometryMap
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
        self._var_geometry = Variables(state_variables={"x": x, "xdot": xdot})
        self._forward_map = ParameterizedGeometryMap(self._var_q, self._fk)

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

    def map(self):
        return self._forward_map
