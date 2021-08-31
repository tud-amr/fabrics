import casadi as ca
import numpy as np
from copy import deepcopy

from optFabrics.diffGeometry.diffMap import DifferentialMap
from optFabrics.diffGeometry.variables import eps
from optFabrics.helper_functions import joinVariables, checkCompatability
from optFabrics.diffGeometry.referenceTrajectory import ReferenceTrajectory


class Spec:
    """description"""

    def __init__(self, M: ca.SX, **kwargs):
        if 'f' in kwargs:
            f = kwargs.get('f')
            assert isinstance(f, ca.SX)
            self._f = f
        if 'h' in kwargs:
            h = kwargs.get('h')
            assert isinstance(h, ca.SX)
            self._h = h
        if 'x' in kwargs:
            self._vars = [kwargs.get('x'), kwargs.get('xdot')]
        elif 'var' in kwargs:
            self._vars = kwargs.get('var')
        self._refTrajs = []
        if 'refTrajs' in kwargs:
            self._refTrajs = kwargs.get('refTrajs')
        self._xdot_d = np.zeros(self.x().size()[0])
        for var in self._vars:
            assert isinstance(var, ca.SX)
        assert isinstance(M, ca.SX)
        self._M = M

    def h(self):
        if hasattr(self, '_h'):
            return self._h
        else:
            return ca.mtimes(self.Minv(), self._f)

    def f(self):
        if hasattr(self, '_f'):
            return self._f
        else:
            return ca.mtimes(self.M(), self._h)

    def M(self):
        return self._M

    def Minv(self):
        import warnings
        warnings.warn("Casadi pseudo inverse is used in weighted geometry")
        return ca.pinv(self._M + np.identity(self.x().size()[0]) * eps)

    def x(self):
        return self._vars[0]

    def xdot(self):
        return self._vars[1]

    def concretize(self):
        self._xddot = -self.h()
        var = deepcopy(self._vars)
        for refTraj in self._refTrajs:
            var += refTraj._vars
        self._funs = ca.Function(
            "M", var, [self.M(), self.f(), self._xddot]
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
        if hasattr(self, '_h') and hasattr(b, '_h') and 1 == 2:
            return Spec(self.M() + b.M(), h=self.h() + b.h(), var=all_vars)
        else:
            return Spec(self.M() + b.M(), f=self.f() + b.f(), var=all_vars)

    def pull(self, dm: DifferentialMap):
        assert isinstance(dm, DifferentialMap)
        M_pulled = ca.mtimes(ca.transpose(dm._J), ca.mtimes(self.M(), dm._J))
        Jt = ca.transpose(dm._J)
        f_1 = ca.mtimes(Jt, ca.mtimes(self.M(), dm.Jdotqdot()))
        f_2 = ca.mtimes(Jt, self.f())
        f_pulled = f_1 + f_2
        M_pulled_subst_x = ca.substitute(M_pulled, self._vars[0], dm._phi)
        M_pulled_subst_x_xdot = ca.substitute(
            M_pulled_subst_x, self._vars[1], dm.phidot()
        )
        f_pulled_subst_x = ca.substitute(f_pulled, self._vars[0], dm._phi)
        f_pulled_subst_x_xdot = ca.substitute(
            f_pulled_subst_x, self._vars[1], dm.phidot()
        )
        var = dm._vars
        if hasattr(dm, '_refTraj'):
            refTrajs = [dm._refTraj] + [refTraj.pull(dm) for refTraj in self._refTrajs]
        else:
            refTrajs = [refTraj.pull(dm) for refTraj in self._refTrajs]
        return Spec(M_pulled_subst_x_xdot, f=f_pulled_subst_x_xdot, var=var, refTrajs=refTrajs)
