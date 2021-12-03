import casadi as ca
import numpy as np

from geomdl import BSpline
from geomdl import utilities
from geomdl.visualization import VisMPL

from optFabrics.diffGeometry.variables import eps
from optFabrics.diffGeometry.diffMap import DifferentialMap
from optFabrics.diffGeometry.symbolicTrajectory import SymbolicTrajectory


class SplineTrajectory(SymbolicTrajectory):

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
        if 'var' in kwargs:
            self._vars = kwargs.get('var')
        else:
            self._vars = [
                ca.SX.sym("x_" + suffix, n), 
                ca.SX.sym("xdot_" + suffix, n), 
                ca.SX.sym("xddot_" + suffix, n)
            ]
        self._vn = 1.0
        if 'vn' in kwargs:
            self._vn = kwargs.get('vn')
        self._lam = 100
        if 'lam' in kwargs:
            self._lam = kwargs.get('lam')
        self._duration = kwargs.get('duration')

    def crv(self):
        return self._crv

    def t(self):
        return self._t

    def concretize(self):
        pass

    def sScaling(self, t):
        return 0.5 * (1 - np.cos(np.pi * t / self._duration))
        return 0.5 * (-np.cos(np.pi * 1/self._duration * t) + 1)

    def vScaling(self, t):
        # return 0.5 * (np.sin(np.pi * t) + 1)
        T = self._duration
        return 1 * np.pi/T * np.sin(t * np.pi/T)

    def aScaling(self, t):
        # return 0.5 * (np.cos(np.pi * t) + 1)
        T = self._duration
        return 1 * (np.pi/T)**2 * np.cos(t * np.pi/T)

    def evaluate(self, t):
        t_ref = self.sScaling(min(t, self._duration))
        xds = self._crv.derivatives(t_ref, order=2)
        x = np.array(xds[0])
        v_raw = np.array(xds[1])
        a_raw = np.array(xds[2])
        v = self.vScaling(t) * v_raw/np.linalg.norm(v_raw)
        a = self.aScaling(t) * a_raw/np.linalg.norm(a_raw)
        if t_ref == 1.0:
            v = v_raw * 0
            a = a_raw * 0
        return x, v, a

    def duration(self):
        return self._duration

    def pull(self, dm: DifferentialMap):
        assert isinstance(dm, DifferentialMap)
        if hasattr(self, '_crv'):
            return SplineTrajectory(self._n, dm._J, t=self.t(), crv=self._crv, var=self._vars, duration=self.duration())
        else:
            return SplineTrajectory(self._n, dm._J, var=self._vars, duration=self.duration())


