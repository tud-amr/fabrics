import casadi as ca
import numpy as np

from geomdl import BSpline
from geomdl import utilities
from geomdl.visualization import VisMPL

from optFabrics.diffGeometry.variables import eps
from optFabrics.diffGeometry.diffMap import DifferentialMap
from optFabrics.diffGeometry.referenceTrajectory import ReferenceTrajectory


class SplineTrajectory(ReferenceTrajectory):

    def __init__(self, n: int, J: ca.SX, **kwargs):
        super().__init__(n, J) 
        if 'crv' in kwargs:
            self._crv = kwargs.get('crv')
        else:
            self._crv = BSpline.Curve()
            self._crv.degree = kwargs.get('degree')
            self._crv.ctrlpts = kwargs.get('ctrlpts')
            self._crv.knotvector = utilities.generate_knot_vector(self._crv.degree, len(self._crv.ctrlpts))
        self._t = kwargs.get('t')
        suffix = "ref"
        if 'name' in kwargs:
            suffix = kwargs.get('name')
        self._x = ca.SX.sym("x_" + suffix, n)
        self._xdot = ca.SX.sym("xdot_" + suffix, n)
        self._xddot = ca.SX.sym("xddot_" + suffix, n)
        self._vn = 1.0
        if 'vn' in kwargs:
            self._vn = kwargs.get('vn')
        self._lam = 100

    def crv(self):
        return self._crv

    def t(self):
        return self._t

    def concretize(self):
        pass

    def vScaling(self, t):
        if t < 0.2:
            return 1/(1 + np.exp(-self._lam * (t - 0.05)))
        elif t > 0.8:
            return 1/(1 + np.exp(self._lam * (t - 0.95)))
        else:
            return 1

    def aScaling(self, t):
        if t < 0.2:
            expTerm = np.exp(-self._lam * (t - 0.05))
            return self._lam * expTerm / (1 + expTerm)**2
        if t > 0.8:
            expTerm = np.exp(self._lam * (t - 0.95))
            return -self._lam * expTerm / (1 + expTerm)**2
        else:
            return 0.0

    def evaluate(self, t):
        xds = self._crv.derivatives(t, order=2)
        x = np.array(xds[0])
        v_raw = np.array(xds[1])
        a_raw = np.array(xds[2])
        v = self.vScaling(t) * v_raw/np.linalg.norm(v_raw)
        a = self.aScaling(t) * a_raw/np.linalg.norm(a_raw)
        return x, v, a

    def x(self):
        return self._x

    def xdot(self):
        return self._xdot

    def xddot(self):
        return self._xddot

    def pull(self, dm: DifferentialMap):
        assert isinstance(dm, DifferentialMap)
        return SplineTrajectory(self._n, dm._J, t=self.t(), crv=self._crv)


