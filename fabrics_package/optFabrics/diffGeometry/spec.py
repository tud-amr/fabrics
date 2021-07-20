import casadi as ca
import numpy as np
import inspect

from optFabrics.diffGeometry.diffMap import DifferentialMap
from optFabrics.diffGeometry.variables import eps


class SpecException(Exception):
    def __init__(self, expression, message):
        self._expression = expression
        self._message = message

    def what(self):
        return self._expression + ": " + self._message


class Spec:
    """description"""

    def __init__(self, M : ca.SX, f : ca.SX, x : ca.SX, xdot : ca.SX):
        assert isinstance(M, ca.SX)
        assert isinstance(f, ca.SX)
        assert isinstance(x, ca.SX)
        assert isinstance(xdot, ca.SX)
        if x.size() != xdot.size():
            raise SpecException(
                "Attempted spec creation failed",
                "Different dimensions of x : "
                + str(b._x.size())
                + " and xdot :"
                + str(self._x.size()),
            )
        self._M = M
        self._f = f
        self._x = x
        self._xdot = xdot

    def concretize(self):
        xddot = ca.mtimes(ca.pinv(self._M + np.identity(self._x.size()[0]) * eps), -self._f)
        self._funs = ca.Function("M", [self._x, self._xdot], [self._M, self._f, xddot])

    def evaluate(self, x : np.ndarray, xdot : np.ndarray):
        assert isinstance(x, np.ndarray)
        assert isinstance(xdot, np.ndarray)
        funs = self._funs(x, xdot)
        M_eval = np.array(funs[0])
        f_eval = np.array(funs[1])[:, 0]
        xddot_eval = np.array(funs[2])[:, 0]
        return [M_eval, f_eval, xddot_eval]

    def __add__(self, b):
        assert isinstance(b, Spec)
        if b._x.size() != self._x.size():
            raise SpecException(
                "Attempted summation invalid",
                "Different dimensions: "
                + str(b._x.size())
                + " vs. "
                + str(self._x.size()),
            )
        if not (ca.is_equal(b._x, self._x)):
            raise SpecException(
                "Attempted summation invalid",
                "Different variables: " + str(b._x) + " vs. " + str(self._x),
            )
        return Spec(self._M + b._M, self._f + b._f, self._x, self._xdot)

    def pull(self, dm : DifferentialMap):
        assert isinstance(dm, DifferentialMap)
        M_pulled = ca.mtimes(ca.transpose(dm._J), ca.mtimes(self._M, dm._J))
        f_1 = ca.mtimes(ca.transpose(dm._J), ca.mtimes(self._M, ca.mtimes(dm._Jdot, dm._qdot)))
        f_2 = ca.mtimes(ca.transpose(dm._J), self._f)
        f_pulled = f_1 + f_2
        M_pulled_subst = ca.substitute(M_pulled, self._x, dm._phi)
        M_pulled_subst2 = ca.substitute(M_pulled_subst, self._xdot, ca.mtimes(dm._J, dm._qdot))
        f_pulled_subst = ca.substitute(f_pulled, self._x, dm._phi)
        f_pulled_subst2 = ca.substitute(f_pulled_subst, self._xdot, ca.mtimes(dm._J, dm._qdot))
        print("spec pull")
        return Spec(M_pulled_subst2, f_pulled_subst2, dm._q, dm._qdot)

