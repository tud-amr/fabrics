import casadi as ca
import numpy as np
from copy import deepcopy

from fabrics.diffGeometry.spec import Spec, checkCompatability
from fabrics.diffGeometry.geometry import Geometry
from fabrics.diffGeometry.energy import Lagrangian
from fabrics.diffGeometry.diffMap import DifferentialMap, DynamicDifferentialMap
from fabrics.diffGeometry.casadi_helpers import outerProduct

from fabrics.helpers.constants import eps
from fabrics.helpers.functions import joinRefTrajs
from fabrics.helpers.casadiFunctionWrapper import CasadiFunctionWrapper

class EnergizedGeometry(Spec):
    # Should not be used as it is not compliant with summation
    # Only used for verification in testing
    def __init__(self, g: Geometry, le: Lagrangian):
        assert isinstance(le, Lagrangian)
        assert isinstance(g, Geometry)
        checkCompatability(le, g)
        frac = outerProduct(g.xdot(), g.xdot()) / (
            eps + ca.dot(g.xdot(), ca.mtimes(le._S._M, g.xdot()))
        )
        pe = np.identity(le.x().size()[0]) - ca.mtimes(le._S._M, frac)
        f = le._S._f + ca.mtimes(pe, ca.mtimes(le._S._M, g._h) - le._S._f)
        super().__init__(le._S._M, f=f, var=g._vars)
        self._le = le


class WeightedGeometry(Spec):
    def __init__(self, **kwargs):
        self._x_ref_name = "x_ref"
        self._xdot_ref_name = "xdot_ref"
        self._xddot_ref_name = "xddot_ref"
        le = kwargs.get("le")
        assert isinstance(le, Lagrangian)
        if 'ref_names' in kwargs:
            ref_names = kwargs.get('ref_names')
            self._x_ref_name = ref_names[0]
            self._xdot_ref_name = ref_names[1]
            self._xddot_ref_name = ref_names[2]
        if "g" in kwargs:
            g = kwargs.get("g")
            checkCompatability(le, g)
            var = g._vars + le._vars
            self._refTrajs = joinRefTrajs(le._refTrajs, g._refTrajs)
            self._le = le
            self._h = g._h
            self._M = le._S.M()
            self._vars = var
        if "s" in kwargs:
            s = kwargs.get("s")
            checkCompatability(le, s)
            self._le = le
            refTrajs = joinRefTrajs(le._refTrajs, s._refTrajs)
            super().__init__(s.M(), f=s.f(), var=s._vars, refTrajs=refTrajs, ref_names=self.ref_names())

    def __add__(self, b):
        spec = super().__add__(b)
        le = self._le + b._le
        return WeightedGeometry(s=spec, le=le, ref_names=spec.ref_names())

    def computeAlpha(self, ref_sign: int = 1):
        xdot = self._le.xdot_rel(ref_sign=ref_sign)
        frac = 1 / (
            eps + ca.dot(xdot, ca.mtimes(self._le._S.M(), xdot))
        )
        self._alpha = -frac * ca.dot(xdot, self.f() - self._le._S.f())

    def concretize(self, ref_sign: int = 1):
        self.computeAlpha(ref_sign=ref_sign)
        self._xddot = -self.h()
        var = deepcopy(self._vars)
        for refTraj in self._refTrajs:
            var += refTraj._vars
        """
        self._funs = ca.Function(
            "M", var, [self.M(), self.f(), self._xddot, self._alpha]
        )
        """
        self._funs = CasadiFunctionWrapper(
                "funs", var.asDict(), {"M": self.M(), 'f': self.f(), 'xddot': self._xddot, 'alpha': self._alpha}
        )

    def evaluate(self, **kwargs):
        evaluations = self._funs.evaluate(**kwargs)
        M = evaluations['M']
        f = evaluations['f']
        xddot = evaluations['xddot']
        alpha = evaluations['alpha']
        return [M, f, xddot, alpha]

    def pull(self, dm: DifferentialMap):
        spec = super().pull(dm)
        le_pulled = self._le.pull(dm)
        return WeightedGeometry(s=spec, le=le_pulled, ref_names=self.ref_names())

    def dynamic_pull(self, dm: DynamicDifferentialMap):
        spec = super().dynamic_pull(dm)
        le_pulled = self._le.dynamic_pull(dm)
        return WeightedGeometry(s=spec, le=le_pulled, ref_names=dm.ref_names())

    def x(self):
        return self._le.x()

    def xdot(self):
        return self._le.xdot()
