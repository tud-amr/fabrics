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
            if 'x_p' in kwargs:
                self._vars += [kwargs.get('x_p'), kwargs.get('xdot_p'), kwargs.get('xddot_p')]
        elif 'var' in kwargs:
            self._vars = kwargs.get('var')
        self._rel = True if len(self._vars) > 2 else False
        self.applyEulerLagrange()

    def x(self):
        return self._vars[0]

    def xdot(self):
        return self._vars[1]

    def x_p(self):
        if self._rel:
            return self._vars[2]
        else:
            return None

    def xdot_p(self):
        if self._rel:
            return self._vars[3]
        else:
            return None

    def xddot_p(self):
        if self._rel:
            return self._vars[4]
        else:
            return None

    def __add__(self, b):
        assert isinstance(b, Lagrangian)
        checkCompatability(self, b)
        return Lagrangian(self._l + b._l, var=joinVariables(self._vars, b._vars))

    def applyEulerLagrange(self):
        dL_dxdot = ca.gradient(self._l, self._vars[1])
        dL_dx = ca.gradient(self._l, self._vars[0])
        d2L_dxdxdot = ca.jacobian(dL_dx, self._vars[1])
        d2L_dxdot2 = ca.jacobian(dL_dxdot, self._vars[1])
        f_rel = np.zeros(self.x().size()[0])
        en_rel = np.zeros(1)

        if self._rel:
            dL_dxpdot = ca.gradient(self._l, self.xdot_p())
            d2L_dxdotdxpdot = ca.jacobian(dL_dxdot, self.xdot_p())
            d2L_dxdotdxp = ca.jacobian(dL_dxdot, self.x_p())
            f_rel1 = ca.mtimes(d2L_dxdotdxpdot, self.xddot_p())
            f_rel2 = ca.mtimes(d2L_dxdotdxp, self.xdot_p())
            f_rel = f_rel1 + f_rel2
            en_rel = ca.dot(dL_dxpdot, self.xdot_p())

        F = d2L_dxdxdot
        f_e = -dL_dx
        M = d2L_dxdot2
        f = ca.mtimes(ca.transpose(F), self._vars[1]) + f_e + f_rel
        self._H = ca.dot(dL_dxdot, self._vars[1]) - self._l + en_rel
        self._S = Spec(M, f=f, var=self._vars)

    def concretize(self):
        self._S.concretize()
        self._l_fun = ca.Function("funs", self._vars, [self._H])

    def evaluate(self, *args):
        for arg in args:
            assert isinstance(arg, np.ndarray)
        l = float(self._l_fun(*args))
        M, f, _ = self._S.evaluate(*args)
        return M, f, l

    def pull(self, dm: DifferentialMap):
        assert isinstance(dm, DifferentialMap)
        l_subst = ca.substitute(self._l, self._vars[0], dm._phi)
        l_subst2 = ca.substitute(l_subst, self._vars[1], dm.phidot())
        new_vars = joinVariables(self._vars[2:], dm._vars)
        return Lagrangian(l_subst2, var=new_vars)


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
