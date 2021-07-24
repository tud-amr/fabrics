import casadi as ca
import numpy as np

from optFabrics.diffGeometry.spec import Spec, checkCompatability
from optFabrics.diffGeometry.diffMap import DifferentialMap


class LagrangianException(Exception):
    def __init__(self, expression, message):
        self._expression = expression
        self._message = message

    def what(self):
        return self._expression + ": " + self._message


class Lagrangian(object):
    """description"""

    def __init__(self, l: ca.SX, **kwargs):
        if len(kwargs) == 2:
            x = kwargs.get('x')
            xdot = kwargs.get('xdot')
        elif len(kwargs) == 1:
            x, xdot = kwargs.get('var')
        assert isinstance(l, ca.SX)
        assert isinstance(x, ca.SX)
        assert isinstance(xdot, ca.SX)
        self._l = l
        self._vars = [x, xdot]
        self.applyEulerLagrange()

    def x(self):
        return self._vars[0]

    def xdot(self):
        return self._vars[1]

    @classmethod
    def fromSpec(cls, l: ca.SX, s: Spec):
        lag = cls(l, var= s._vars)
        lag._S = s
        return lag

    def __add__(self, b):
        assert isinstance(b, Lagrangian)
        checkCompatability(self, b)
        return Lagrangian.fromSpec(self._l + b._l, self._S + b._S)

    def applyEulerLagrange(self):
        dL_dx = ca.gradient(self._l, self.x())
        dL_dxdot = ca.gradient(self._l, self.xdot())
        d2L_dxdxdot = ca.jacobian(dL_dx, self.xdot())
        d2L_dxdot2 = ca.jacobian(dL_dxdot, self.xdot())

        F = d2L_dxdxdot
        f_e = -dL_dx
        M = d2L_dxdot2
        f = ca.mtimes(ca.transpose(F), self.xdot()) + f_e
        self._S = Spec(M, f, var=self._vars)

    def concretize(self):
        self._S.concretize()
        self._l_fun = ca.Function("funs", self._vars, [self._l])

    def evaluate(self, x: np.ndarray, xdot: np.ndarray):
        assert isinstance(x, np.ndarray)
        assert isinstance(xdot, np.ndarray)
        l = float(self._l_fun(x, xdot))
        M, f, _ = self._S.evaluate(x, xdot)
        return M, f, l

    def pull(self, dm: DifferentialMap):
        assert isinstance(dm, DifferentialMap)
        l_subst = ca.substitute(self._l, self.x(), dm._phi)
        l_pulled = ca.substitute(l_subst, self.xdot(), ca.mtimes(dm._J, dm.qdot()))
        s_pulled = self._S.pull(dm)
        return Lagrangian.fromSpec(l_pulled, s_pulled)


class FinslerStructure(Lagrangian):
    def __init__(self, lg: ca.SX, **kwargs):
        self._lg = lg
        l = 0.5 * lg ** 2
        super().__init__(l, **kwargs)

    def concretize(self):
        super().concretize()
        self._lg_fun = ca.Function("fun_lg", self._vars, [self._lg])

    def evaluate(self, x: np.ndarray, xdot: np.ndarray):
        M, f, l = super().evaluate(x, xdot)
        lg = float(self._lg_fun(x, xdot))
        return M, f, l, lg
