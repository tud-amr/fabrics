import casadi as ca
import numpy as np

from optFabrics.diffGeometry.spec import Spec, checkCompatability
from optFabrics.diffGeometry.geometry import Geometry
from optFabrics.diffGeometry.energy import Lagrangian
from optFabrics.diffGeometry.diffMap import DifferentialMap, RelativeDifferentialMap
from optFabrics.diffGeometry.casadi_helpers import outerProduct
from optFabrics.diffGeometry.variables import eps

from optFabrics.helper_functions import joinVariables

class EnergizedGeometry(Spec):
    # Should not be used as it is not compliant with summation
    def __init__(self, g: Geometry, le: Lagrangian):
        assert isinstance(le, Lagrangian)
        assert isinstance(g, Geometry)
        checkCompatability(le, g)
        frac = outerProduct(g.xdot(), g.xdot()) / (
            eps + ca.dot(g.xdot(), ca.mtimes(le._S._M, g.xdot()))
        )
        pe = np.identity(le.x().size()[0]) - ca.mtimes(le._S._M, frac)
        f = le._S._f + ca.mtimes(pe, ca.mtimes(le._S._M, g._h) - le._S._f)
        super().__init__(le._S._M, f, var=g._vars)
        self._le = le


class WeightedGeometry(Spec):
    def __init__(self, **kwargs):
        le = kwargs.get("le")
        assert isinstance(le, Lagrangian)
        if "g" in kwargs:
            g = kwargs.get("g")
            checkCompatability(le, g)
            var = joinVariables(g._vars, le._vars)
            self._le = le
            super().__init__(le._S._M, ca.mtimes(le._S._M, g._h), var=var)
        if "s" in kwargs:
            s = kwargs.get("s")
            checkCompatability(le, s)
            self._le = le
            super().__init__(s._M, s._f, var=s._vars)

    def __add__(self, b):
        spec = super().__add__(b)
        le = self._le + b._le
        return WeightedGeometry(s=spec, le=le)

    def computeAlpha(self):
        frac = 1 / (
            eps + ca.dot(self.xdot(), ca.mtimes(self._le._S._M, self.xdot()))
        )
        self._alpha = -frac * ca.dot(self.xdot(), self._f - self._le._S._f)

    def concretize(self):
        self.computeAlpha()
        self._xddot = ca.mtimes(
            ca.pinv(self._M + np.identity(self.x().size()[0]) * eps), -self._f
        )
        self._funs = ca.Function(
            "M", self._vars, [self._M, self._f, self._xddot, self._alpha]
        )

    def evaluate(self, *args):
        for arg in args:
            assert isinstance(arg, np.ndarray)
        funs = self._funs(*args)
        M_eval = np.array(funs[0])
        f_eval = np.array(funs[1])[:, 0]
        xddot_eval = np.array(funs[2])[:, 0]
        alpha_eval = float(funs[3])
        return [M_eval, f_eval, xddot_eval, alpha_eval]

    def pull(self, dm: DifferentialMap):
        spec = super().pull(dm)
        le_pulled = self._le.pull(dm)
        return WeightedGeometry(s=spec, le=le_pulled)

    def x(self):
        return self._le.x()

    def xdot(self):
        return self._le.xdot()
