import numpy as np
from scipy.integrate import odeint
import casadi as ca

from optFabrics.leaf import ForcingLeaf, DynamicLeaf
from optFabrics.functions import generateLagrangian, generateHamiltonian

class RootGeometry(object):
    def __init__(self, leaves, le, n, damper=None, he_fun=None):
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
        self._he_fun = he_fun
        self._le_fun = ca.Function("le", [q, qdot], [le])
        self.M_base_fun = ca.Function("M_base", [q, qdot], [M_base])
        self._M_geometry = np.zeros((n, n))
        self._M_forcing = np.zeros((n, n))
        self._M_aug = np.identity(2 * n)
        self._damper = damper
        self._d = np.zeros(n)

    def update(self, q, qdot, t):
        self._q = q
        self._qdot = qdot
        self._M_geometry = self.M_base_fun(q, qdot)
        self._M_forcing = np.zeros((self._n, self._n))
        self._f_geometry = np.zeros(self._n)
        self._f_forcing = np.zeros(self._n)
        self._fe_geometry = np.zeros(self._n)
        self._fe_forcing = np.zeros(self._n)
        x_forcing = []
        he = 0.0
        if self._he_fun:
            he = self._he_fun(q[0], qdot[0])
        for leaf in self._leaves:
            isForcing = isinstance(leaf, ForcingLeaf) or isinstance(leaf, DynamicLeaf)
            (M_leaf, f_leaf, fe_leaf, he_leaf) = leaf.pull(q, qdot, t)
            if isForcing:
                self._M_forcing += M_leaf
                self._f_forcing += f_leaf
                self._fe_forcing += fe_leaf
                x_leaf, _, _, _, _ = leaf._diffMap.forwardMap(q, qdot, t)
                x_forcing += x_leaf.tolist()
            else:
                self._M_geometry += M_leaf
                self._f_geometry += f_leaf
                self._fe_geometry += fe_leaf
            he += he_leaf
        if self._damper:
            x_forcing = np.array(x_forcing)
            (alex, beta) = self._damper.damp(self._f_geometry, self._f_forcing, self._fe_geometry, self._fe_forcing, self._M_geometry, self._M_forcing, q, qdot, x_forcing)
            #alex = 0.0
            #beta = 100.0
            self._d = (beta - alex) * np.dot((self._M_forcing + self._M_geometry), qdot)
            for leaf in self._leaves:
                if isinstance(leaf, DynamicLeaf):
                    self._d += leaf.bxddot(beta)
        self._he = he

    def setRHS(self):
        self._rhs = -self._f_forcing - self._f_geometry - self._d

    def augment(self):
        n = self._n
        for i in range(n):
            self._rhs_aug[i] = self._qdot[i]
            self._rhs_aug[i + n] = self._rhs[i]
        M = self._M_forcing + self._M_geometry
        if np.linalg.cond(M) > 1e5:
            M += np.identity(n) * 0.01
        self._M_aug[n:2*n, n:2*n] = M

    def contDynamics(self, z, t):
        self.update(z[0:self._n], z[self._n:2*self._n], t)
        self.setRHS()
        self.augment()
        zdot = np.dot(np.linalg.pinv(self._M_aug), self._rhs_aug)
        return zdot

    def le(self):
        return self._he

    def computePath(self, z0, dt, T):
        t = np.arange(0.0, T, step=dt)
        sol, info = odeint(self.contDynamics, z0, t, full_output=True)
        return sol
