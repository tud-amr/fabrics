import casadi as ca
import numpy as np
import inspect

from optFabrics.diffGeometry.diffMap import DifferentialMap
from optFabrics.diffGeometry.variables import eps
from optFabrics.helper_functions import joinVariables, checkCompatability
from optFabrics.exceptions.spec_exception import SpecException


class Spec:
    """description"""

    def __init__(self, M: ca.SX, f: ca.SX, **kwargs):
        if 'x' in kwargs:
            self._vars = [kwargs.get('x'), kwargs.get('xdot')]
        elif 'var' in kwargs:
            self._vars = kwargs.get('var')
        self._xdot_d = np.zeros(self.x().size()[0])
        for var in self._vars:
            assert isinstance(var, ca.SX)
        assert isinstance(M, ca.SX)
        assert isinstance(f, ca.SX)
        self._M = M
        self._f = f

    def x(self):
        return self._vars[0]

    def xdot(self):
        return self._vars[1]

    def concretize(self):
        self._xddot = ca.mtimes(
            ca.pinv(self._M + np.identity(self.x().size()[0]) * eps), -self._f
        )
        self._funs = ca.Function(
            "M", self._vars, [self._M, self._f, self._xddot]
        )

    def evaluate(self, *args):
        for arg in args:
            assert isinstance(arg, np.ndarray)
        funs = self._funs(*args)
        M_eval = np.array(funs[0])
        f_eval = np.array(funs[1])[:, 0]
        xddot_eval = np.array(funs[2])[:, 0]
        return [M_eval, f_eval, xddot_eval]

    def __add__(self, b):
        assert isinstance(b, Spec)
        checkCompatability(self, b)
        all_vars = joinVariables(self._vars, b._vars)
        return Spec(self._M + b._M, self._f + b._f, var=all_vars)

    def pull(self, dm: DifferentialMap):
        assert isinstance(dm, DifferentialMap)
        M_pulled = ca.mtimes(ca.transpose(dm._J), ca.mtimes(self._M, dm._J))
        Jt = ca.transpose(dm._J)
        f_1 = ca.mtimes(Jt, ca.mtimes(self._M, dm.Jdotqdot()))
        f_2 = ca.mtimes(Jt, self._f)
        f_pulled = f_1 + f_2
        M_pulled_subst_x = ca.substitute(M_pulled, self.x(), dm._phi)
        M_pulled_subst_x_xdot = ca.substitute(
            M_pulled_subst_x, self.xdot(), ca.mtimes(dm._J, dm.qdot())
        )
        f_pulled_subst_x = ca.substitute(f_pulled, self.x(), dm._phi)
        f_pulled_subst_x_xdot = ca.substitute(
            f_pulled_subst_x, self.xdot(), dm.phidot()
        )
        var = joinVariables(dm._vars, self._vars[2:])
        return Spec(M_pulled_subst_x_xdot, f_pulled_subst_x_xdot, var=var)
