import casadi as ca
import numpy as np

from fabrics.helpers.casadiFunctionWrapper import CasadiFunctionWrapper
from fabrics.helpers.variables import Variables


class DifferentialMap:
    _vars: Variables
    _J: ca.SX
    _Jdot: ca.SX

    def __init__(self, phi: ca.SX, variables: Variables, **kwargs):
        assert isinstance(phi, ca.SX)
        assert isinstance(variables, Variables)
        self._vars = variables
        Jdot_sign = -1
        if 'Jdot_sign' in kwargs.keys():
            Jdot_sign = kwargs.get('Jdot_sign')
        self._vars.verify()
        self._phi = phi
        q = self._vars.position_variable()
        qdot = self._vars.velocity_variable()
        self._J = ca.jacobian(phi, q)
        self._Jdot = Jdot_sign * ca.jacobian(ca.mtimes(self._J, qdot), q)

    def Jdotqdot(self) -> ca.SX:
        return ca.mtimes(self._Jdot, self.qdot())

    def phidot(self) -> ca.SX:
        return ca.mtimes(self._J, self.qdot())

    def concretize(self) -> None:
        self._funs = CasadiFunctionWrapper(
            "funs", self._vars.asDict(), {"phi": self._phi, "J": self._J, "Jdot": self._Jdot}
        )

    def params(self) -> dict:
        return self._vars.parameters()

    def state_variables(self) -> dict:
        return self._vars.state_variables()

    def forward(self, **kwargs):
        evaluations = self._funs.evaluate(**kwargs)
        x = evaluations['phi']
        J = evaluations['J']
        Jdot = evaluations['Jdot']
        return x, J, Jdot

    def q(self):
        return self._vars.position_variable()

    def qdot(self):
        return self._vars.velocity_variable()

class DynamicDifferentialMap(DifferentialMap):
    _phi_dot: ca.SX
    _Jdotqdot: ca.SX

    def __init__(self, variables: Variables, ref_names=['x_ref', 'xdot_ref', 'xddot_ref'], **kwargs):
        self._x_ref_name = ref_names[0]
        self._xdot_ref_name = ref_names[1]
        self._xddot_ref_name = ref_names[2]
        phi = variables.position_variable() - variables.parameter_by_name(self._x_ref_name)
        self._phi_dot = variables.velocity_variable() - variables.parameter_by_name(self._xdot_ref_name)

        super().__init__(phi, variables, **kwargs)

    def x_ref(self) -> ca.SX:
        return self._vars.parameter_by_name(self._x_ref_name)

    def xdot_ref(self) -> ca.SX:
        return self._vars.parameter_by_name(self._xdot_ref_name)

    def xddot_ref(self) -> ca.SX:
        return self._vars.parameter_by_name(self._xddot_ref_name)

    def ref_names(self) -> list:
        return [self._x_ref_name, self._xdot_ref_name, self._xddot_ref_name]

    def phidot(self) -> ca.SX:
        return self._phi_dot

    def concretize(self) -> None:
        self._funs = CasadiFunctionWrapper(
            "funs", self._vars.asDict(), {"x_rel": self._phi, "xdot_rel": self._phi_dot}
        )

    def forward(self, **kwargs):
        evaluations = self._funs.evaluate(**kwargs)
        x = evaluations['x_rel']
        xdot = evaluations['xdot_rel']
        return x, xdot

class ExplicitDifferentialMap(DifferentialMap):
    """Explicit differential map for which the gradients can be computed at runtime.

    This class is a special differential map for which the Jacobian matrices
    can be set numerically at runtime.
    """
    def __init__(self, phi: ca.SX, variables: Variables, **kwargs):
        super().__init__(phi, variables, **kwargs)
        try:
            self._J = kwargs.get("J")
            self._Jdot = kwargs.get("Jdot")
        except Exception as e:
            raise Exception("J and Jdot not defined for ExplicitDifferentialMap")

