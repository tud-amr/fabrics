import casadi as ca
from fabrics.helpers.variables import Variables


class Leaf(object):
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
        leaf_variables = Variables(
            state_variables={f"x_{leaf_name}": self._x, f"xdot_{leaf_name}": self._xdot}
        )
        self._leaf_variables = leaf_variables
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
