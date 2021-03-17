import numpy as np
from scipy.integrate import odeint
import casadi as ca

from optFabrics.leaf import ForcingLeaf
from optFabrics.functions import generateLagrangian

class RootGeometry(object):
    def __init__(self, leaves, le, n, damper=None):
        self._n = n
        self._leaves = leaves
        self._f_geometry = np.zeros(n)
        self._f_forcing = np.zeros(n)
        self._fe_geometry = np.zeros(n)
        self._fe_forcing = np.zeros(n)
        self._rhs = np.zeros(n)
        self._rhs_aug = np.zeros(2*n)
        self._q = np.zeros(n)
        self._qdot = np.zeros(n)
        q = ca.SX.sym('q', n)
        qdot = ca.SX.sym('qdot', n)
        M_base, _ = generateLagrangian(le, q, qdot, "base")
        self.M_base_fun = ca.Function("M_base", [q, qdot], [M_base])
        self._M_geometry = np.zeros((n, n))
        self._M_forcing = np.zeros((n, n))
        self._M_aug = np.identity(2 * n)
        self._damper = damper
        self._d = np.zeros(n)

    def update(self, q, qdot, t=None):
        self._q = q
        self._qdot = qdot
        self._M_geometry = self.M_base_fun(q, qdot)
        self._M_forcing = np.zeros((self._n, self._n))
        self._f_geometry = np.zeros(self._n)
        self._f_forcing = np.zeros(self._n)
        self._fe_geometry = np.zeros(self._n)
        self._fe_forcing = np.zeros(self._n)
        for leaf in self._leaves:
            (M_leaf, f_leaf, fe_leaf) = leaf.pull(q, qdot, t)
            isForcing = isinstance(leaf, ForcingLeaf)
            if isForcing:
                self._M_forcing += M_leaf
                self._f_forcing += f_leaf
                self._fe_forcing += fe_leaf
                x, _, _, _, _ = leaf._diffMap.forwardMap(q, qdot)
            else:
                self._M_geometry += M_leaf
                self._f_geometry += f_leaf
                self._fe_geometry += fe_leaf
        if self._damper:
            (alex, beta) = self._damper.damp(self._f_geometry, self._f_forcing, self._fe_geometry, self._fe_forcing, self._M_geometry, self._M_forcing, q, qdot, x)
            self._d = (beta - alex) * np.dot((self._M_forcing + self._M_geometry), qdot)

    def setRHS(self):
        self._rhs = -self._f_forcing - self._f_geometry - self._d

    def augment(self):
        n = self._n
        for i in range(n):
            self._rhs_aug[i] = self._qdot[i]
            self._rhs_aug[i + n] = self._rhs[i]
        self._M_aug[n:2*n, n:2*n] = self._M_forcing + self._M_geometry

    def contDynamics(self, z, t):
        self.update(z[0:self._n], z[self._n:2*self._n], t)
        self.setRHS()
        self.augment()
        zdot = np.dot(np.linalg.pinv(self._M_aug), self._rhs_aug)
        return zdot

    def computePath(self, z0, dt, T):
        t = np.arange(0.0, T, step=dt)
        sol, info = odeint(self.contDynamics, z0, t, full_output=True)
        return sol
