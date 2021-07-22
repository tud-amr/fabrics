import casadi as ca
import numpy as np

from optFabrics.diffGeometry.diffMap import DifferentialMap
from optFabrics.diffGeometry.variables import eps


class Geometry:
    """description"""

    def __init__(self, **kwargs):
        if len(kwargs) == 3:
            h = kwargs.get("h")
            x = kwargs.get("x")
            xdot = kwargs.get("xdot")
            assert isinstance(h, ca.SX)
            assert isinstance(x, ca.SX)
            assert isinstance(xdot, ca.SX)
            self._h = h
            self._x = x
            self._xdot = xdot

        if len(kwargs) == 1:
            s = kwargs.get("s")
            M_eps = s._M + np.identity(s._x.size()[0]) * eps
            self._h = ca.mtimes(ca.pinv(M_eps), s._f)
            self._x = s._x
            self._xdot = s._xdot

    def __add__(self, b):
        assert isinstance(b, Geometry)
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
        return Geometry(h=self._h + b._h, x=self._x, xdot=self._xdot)

    def pull(self, dm: DifferentialMap):
        assert isinstance(dm, DifferentialMap)
        Jt = ca.transpose(dm._J)
        JtJ = ca.mtimes(Jt, dm._J)
        h_1 = ca.mtimes(Jt, ca.mtimes(dm._Jdot, dm._qdot))
        h_2 = ca.mtimes(Jt, self._h)
        JtJ_eps = JtJ + np.identity(dm._q.size()[0]) * eps
        h_pulled = ca.mtimes(ca.pinv(JtJ_eps), h_1 + h_2)
        h_pulled_subst = ca.substitute(h_pulled, self._x, dm._phi)
        h_pulled_subst2 = ca.substitute(
            h_pulled_subst, self._xdot, ca.mtimes(dm._J, dm._qdot)
        )
        return Geometry(h=h_pulled_subst2, x=dm._q, xdot=dm._qdot)

    def concretize(self):
        xddot = -self._h
        self._funs = ca.Function("funs", [self._x, self._xdot], [self._h, xddot])

    def evaluate(self, x: np.ndarray, xdot: np.ndarray):
        assert isinstance(x, np.ndarray)
        assert isinstance(xdot, np.ndarray)
        funs = self._funs(x, xdot)
        h_eval = np.array(funs[0])[:, 0]
        xddot_eval = np.array(funs[1])[:, 0]
        return [h_eval, xddot_eval]

    def testHomogeneousDegree2(self):
        x = np.random.rand(self._x.size()[0])
        xdot = np.random.rand(self._xdot.size()[0])
        alpha = 2.0
        xdot2 = alpha * xdot
        h, _ = self.evaluate(x, xdot)
        h2, _ = self.evaluate(x, xdot2)
        return h * alpha ** 2 == h2
