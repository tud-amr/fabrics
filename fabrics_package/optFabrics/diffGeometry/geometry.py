import casadi as ca
import numpy as np

from optFabrics.diffGeometry.spec import Spec, SpecException
from optFabrics.diffGeometry.energy import Lagrangian, FinslerStructure
from optFabrics.diffGeometry.casadi_helpers import outerProduct
from optFabrics.diffGeometry.diffMap import DifferentialMap

class Geometry(Spec):
    """description"""

    def __init__(self, h : ca.SX, x : ca.SX, xdot : ca.SX):
        M = ca.SX(np.identity(x.size()[0]))
        super().__init__(M, h, x, xdot)


    def energize(self, le : Lagrangian):
        assert isinstance(le, Lagrangian)
        if le._x.size() != self._x.size():
            raise SpecException(
                "Attempted energization invalid",
                "Different dimensions: "
                + str(le._x.size())
                + " vs. "
                + str(self._x.size()),
            )
        if not (ca.is_equal(le._x, self._x)):
            raise SpecException(
                "Attempted energization invalid",
                "Different variables: " + str(le._x) + " vs. " + str(self._x),
            )
        eps = 1e-8
        frac =outerProduct(self._xdot, self._xdot)/(eps + ca.dot(self._xdot, ca.mtimes(le._S._M, self._xdot)))
        pe = np.identity(le._x.size()[0]) - ca.mtimes(le._S._M, frac)
        f = le._S._f + ca.mtimes(pe, ca.mtimes(le._S._M, self._f) - le._S._f)
        alpha = self.alpha(le)
        return EnergizedGeometry(le._S._M, f, self._x, self._xdot, alpha, le)

    def alpha(self, le : Lagrangian):
        assert isinstance(le, Lagrangian)
        eps = 1e-8
        frac = self._xdot/(eps + ca.dot(self._xdot, ca.mtimes(le._S._M, self._xdot)))
        return ca.dot(frac, le._S._f - self._f)


    def testHomogeneousDegree2(self):
        x = np.random.rand(self._x.size()[0])
        xdot = np.random.rand(self._xdot.size()[0])
        alpha = 2.0
        xdot2 = alpha * xdot
        _, h, _ = self.evaluate(x, xdot)
        _, h2, _ = self.evaluate(x, xdot2)
        return h * alpha**2 == h2

class EnergizedGeometry(Spec):

    def __init__(self, M : ca.SX, f : ca.SX, x : ca.SX, xdot : ca.SX, alpha : ca.SX, le : Lagrangian):
        self._alpha = alpha
        self._le = le
        super().__init__(M, f, x, xdot)

    @classmethod
    def fromSpec(cls, s : Spec, alpha : ca.SX, le : ca.SX):
        assert isinstance(s, Spec)
        assert isinstance(alpha, ca.SX)
        assert isinstance(le, Lagrangian)
        return cls(s._M, s._f, s._x, s._xdot, alpha, le)

    def concretize(self):
        xddot = ca.mtimes(ca.pinv(self._M), -self._f)
        self._funs = ca.Function("M", [self._x, self._xdot], [self._M, self._f, xddot, self._alpha])

    def evaluate(self, x : np.ndarray, xdot : np.ndarray):
        assert isinstance(x, np.ndarray)
        assert isinstance(xdot, np.ndarray)
        funs = self._funs(x, xdot)
        M_eval = np.array(funs[0])
        f_eval = np.array(funs[1])[:, 0]
        xddot_eval = np.array(funs[2])[:, 0]
        alpha_eval = float(funs[3])
        return [M_eval, f_eval, xddot_eval, alpha_eval]

    def pull(self, dm : DifferentialMap):
        spec = super().pull(dm)
        alpha_subst = ca.substitute(self._alpha, self._x, dm._phi)
        alpha_subst2 = ca.substitute(alpha_subst, self._xdot, ca.mtimes(dm._J, dm._qdot))
        le_pulled = self._le.pull(dm)
        return EnergizedGeometry.fromSpec(spec, alpha_subst2, le_pulled)

