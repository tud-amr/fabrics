import casadi as ca
from fabrics.diffGeometry.diffMap import DynamicParameterizedDifferentialMap
from fabrics.helpers.variables import Variables


class DynamicLeaf(object):
    def __init__(
        self,
        parent_variables: Variables,
        leaf_name: str,
        forward_kinematics: ca.SX,
        dim: int = 1,
    ):
        self._parent_variables = parent_variables
        self._x = ca.SX.sym(f"x_{leaf_name}", dim)
        self._xdot = ca.SX.sym(f"xdot_{leaf_name}", dim)
        self._x_ref = ca.SX.sym(f"x_ref_{leaf_name}", dim)
        self._xdot_ref = ca.SX.sym(f"xdot_ref_{leaf_name}", dim)
        self._xddot_ref = ca.SX.sym(f"xddot_ref_{leaf_name}", dim)
        reference_parameters = {
            f"x_ref_{leaf_name}": self._x_ref,
            f"xdot_ref_{leaf_name}": self._xdot_ref,
            f"xddot_ref_{leaf_name}": self._xddot_ref,
        }
        leaf_variables = Variables(
            state_variables={f"x_{leaf_name}": self._x, f"xdot_{leaf_name}": self._xdot},
            parameters=reference_parameters
        )
        phi_dynamic = self._x - self._x_ref
        phi_dot_dynamic = self._xdot - self._xdot_ref
        Jdotqdot_dynamic = -self._xddot_ref
        self._dynamic_map = DynamicParameterizedDifferentialMap(
                phi_dynamic, phi_dot_dynamic, Jdotqdot_dynamic, var=leaf_variables)
        self._parent_variables.add_parameters(reference_parameters)
        self._leaf_variables = leaf_variables
        self._forward_kinematics = forward_kinematics
        self._p = {}
        self._dm = None
        self._lag = None
        self._geo = None

    def set_params(self, **kwargs):
        for key in self._p:
            if key in kwargs:
                self._p[key] = kwargs.get(key)

    def geometry(self):
        return self._geo

    def map(self):
        return self._dm

    def lagrangian(self):
        return self._lag

    def dynamic_map(self):
        return self._dynamic_map
