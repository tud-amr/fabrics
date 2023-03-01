import casadi as ca
from fabrics.diffGeometry.diffMap import DynamicDifferentialMap
from fabrics.helpers.variables import Variables


class DynamicLeaf(object):
    def __init__(
        self,
        parent_variables: Variables,
        leaf_name: str,
        forward_kinematics: ca.SX,
        dim: int = 1,
        dim_ref: int = 1,
        reference_parameters: dict = None,
    ):
        self._dim_ref = dim_ref
        self._dim = dim
        self._parent_variables = parent_variables
        self._x = ca.SX.sym(f"x_{leaf_name}", dim)
        self._xdot = ca.SX.sym(f"xdot_{leaf_name}", dim)
        self._fk_x = ca.SX.sym(f"fk_x_{leaf_name}", dim_ref)
        self._fk_xdot = ca.SX.sym(f"fk_xdot_{leaf_name}", dim_ref)
        self._x_rel = ca.SX.sym(f"x_rel_{leaf_name}", dim_ref)
        self._xdot_rel = ca.SX.sym(f"xdot_rel_{leaf_name}", dim_ref)
        if not reference_parameters:
            self._x_ref = ca.SX.sym(f"x_ref_{leaf_name}", dim_ref)
            self._xdot_ref = ca.SX.sym(f"xdot_ref_{leaf_name}", dim_ref)
            self._xddot_ref = ca.SX.sym(f"xddot_ref_{leaf_name}", dim_ref)
            reference_parameters = {
                f"x_ref_{leaf_name}": self._x_ref,
                f"xdot_ref_{leaf_name}": self._xdot_ref,
                f"xddot_ref_{leaf_name}": self._xddot_ref,
            }
        else:
            self._x_ref = list(reference_parameters.values())[0]
            self._xdot_ref = list(reference_parameters.values())[1]
            self._xddot_ref = list(reference_parameters.values())[2]
        self._fk_variables = Variables(
            state_variables={f"fk_x_{leaf_name}": self._fk_x, f"fk_xdot_{leaf_name}": self._fk_xdot},
            parameters=reference_parameters,
        )
        self._leaf_variables = Variables(
            state_variables={f"x_{leaf_name}": self._x, f"xdot_{leaf_name}": self._xdot},
        )
        self._relative_variables = Variables(
                state_variables={'x_rel': self._x_rel, 'xdot_rel': self._xdot_rel}
            )
        phi_dynamic = self._x - self._x_ref
        phi_dot_dynamic = self._xdot - self._xdot_ref
        Jdotqdot_dynamic = -self._xddot_ref
        self._dynamic_map = DynamicDifferentialMap(
                self._fk_variables, ref_names=list(reference_parameters.keys())
        )
        self._parent_variables.add_parameters(reference_parameters)
        self._forward_kinematics = forward_kinematics
        self._p = {}
        self._dm = None
        self._lag = None
        self._geo = None
        self._leaf_name = leaf_name

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
