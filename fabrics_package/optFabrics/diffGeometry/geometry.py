import casadi as ca
import numpy as np

from optFabrics.diffGeometry.diffMap import DifferentialMap
from optFabrics.diffGeometry.variables import eps
from optFabrics.helper_functions import joinVariables


class Geometry:
    """description"""

    def __init__(self, **kwargs):
        if 'x' in kwargs:
            h = kwargs.get("h")
            x = kwargs.get("x")
            xdot = kwargs.get("xdot")
            self._vars = [x, xdot]
        elif 'var' in kwargs:
            h = kwargs.get("h")
            self._vars = kwargs.get('var')
        elif 's' in kwargs:
            s = kwargs.get("s")
            h = s.h()
            self._vars = s._vars

        assert isinstance(h, ca.SX)
        self._h = h

    def x(self):
        return self._vars[0]

    def xdot(self):
        return self._vars[1]

    def __add__(self, b):
        assert isinstance(b, Geometry)
        # TODO: checkCompatibility()  <24-07-21, mspahn> #
        nb_vars_a = len(self._vars)
        nb_vars_b = len(b._vars)
        if nb_vars_a >= nb_vars_b:
            var = self._vars
        else:
            var = b._vars
        return Geometry(h=self._h + b._h, var=var)

    def pull(self, dm: DifferentialMap):
        assert isinstance(dm, DifferentialMap)
        Jt = ca.transpose(dm._J)
        JtJ = ca.mtimes(Jt, dm._J)
        h_1 = ca.mtimes(Jt, self._h)
        h_2 = ca.mtimes(Jt, dm.Jdotqdot())
        JtJ_eps = JtJ + np.identity(dm.q().size()[0]) * eps
        h_pulled = ca.mtimes(ca.pinv(JtJ_eps), h_1 + h_2)
        h_pulled_subst_x = ca.substitute(h_pulled, self.x(), dm._phi)
        h_pulled_subst_x_xdot = ca.substitute(h_pulled_subst_x, self.xdot(), dm.phidot())
        var = joinVariables(dm._vars, self._vars[2:])
        return Geometry(h=h_pulled_subst_x_xdot, var=var)

    def concretize(self):
        self._xddot = -self._h
        self._funs = ca.Function("funs", self._vars, [self._h, self._xddot])

    def evaluate(self, *args):
        funs = self._funs(*args)
        h_eval = np.array(funs[0])[:, 0]
        xddot_eval = np.array(funs[1])[:, 0]
        return [h_eval, xddot_eval]

    def testHomogeneousDegree2(self):
        x = np.random.rand(self.x().size()[0])
        xdot = np.random.rand(self.xdot().size()[0])
        alpha = 2.0
        xdot2 = alpha * xdot
        h, _ = self.evaluate(x, xdot)
        h2, _ = self.evaluate(x, xdot2)
        return h * alpha ** 2 == h2
