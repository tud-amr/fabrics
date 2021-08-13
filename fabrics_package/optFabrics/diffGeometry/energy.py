import casadi as ca
import numpy as np

from optFabrics.diffGeometry.spec import Spec, checkCompatability
from optFabrics.diffGeometry.diffMap import DifferentialMap, RelativeDifferentialMap

from optFabrics.helper_functions import joinVariables


class LagrangianException(Exception):
    def __init__(self, expression, message):
        self._expression = expression
        self._message = message

    def what(self):
        return self._expression + ": " + self._message


class Lagrangian(object):
    """description"""

    def __init__(self, l: ca.SX, **kwargs):
        self._l = l
        assert isinstance(l, ca.SX)
        if 'x' in kwargs:
            self._vars = [kwargs.get('x'), kwargs.get('xdot')]
            self.applyEulerLagrange()
        elif 'var' in kwargs:
            self._vars = kwargs.get('var')
            self.applyEulerLagrange()
        if 's' in kwargs:
            s = kwargs.get('s')
            self._vars = s._vars
            self._S = s

    def x(self):
        return self._vars[0]

    def xdot(self):
        return self._vars[1]

    def __add__(self, b):
        assert isinstance(b, Lagrangian)
        checkCompatability(self, b)
        return Lagrangian(self._l + b._l, s=self._S + b._S)

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

    def evaluate(self, *args):
        for arg in args:
            assert isinstance(arg, np.ndarray)
        l = float(self._l_fun(*args))
        M, f, _ = self._S.evaluate(*args)
        return M, f, l

    def pull(self, dm: DifferentialMap):
        assert isinstance(dm, DifferentialMap)
        l_subst = ca.substitute(self._l, self._vars[0], dm._phi)
        l_pulled = ca.substitute(l_subst, self._vars[1], dm.phidot())
        s_pulled = self._S.pull(dm)
        if isinstance(dm, RelativeDifferentialMap):
            return RelativeLagrangian(l_pulled, dm=dm, s=s_pulled)
        else:
            return Lagrangian(l_pulled, s=s_pulled)


class RelativeLagrangian(Lagrangian):

    def __init__(self, l: ca.SX, **kwargs):
        self._dm = kwargs.get('dm')
        super().__init__(l, **kwargs)
        self._vars = joinVariables(self._dm._vars, self._vars)

    def x(self):
        return self._vars[0]

    def xdot(self):
        return (self._vars[1] - self._dm.qdot_p())


class FinslerStructure(Lagrangian):
    def __init__(self, lg: ca.SX, **kwargs):
        self._lg = lg
        l = 0.5 * lg ** 2
        super().__init__(l, **kwargs)

    def concretize(self):
        super().concretize()
        self._lg_fun = ca.Function("fun_lg", self._vars, [self._lg])

    def evaluate(self, *args):
        M, f, l = super().evaluate(*args)
        lg = float(self._lg_fun(*args))
        return M, f, l, lg
