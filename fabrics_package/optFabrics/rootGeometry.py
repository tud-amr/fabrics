import numpy as np
from scipy.integrate import odeint

class RootGeometry(object):
    def __init__(self, leaves, n, damper=None):
        self._n = n
        self._leaves = leaves
        self._h = np.zeros(n)
        self._rhs = np.zeros(n)
        self._rhs_aug = np.zeros(2*n)
        self._q = np.zeros(n)
        self._qdot = np.zeros(n)
        self._M = np.zeros((n, n))
        self._damper = damper
        self._d = np.zeros(n)

    def update(self, q, qdot):
        self._q = q
        self._qdot = qdot
        self._M = np.zeros((self._n, self._n))
        h_int = np.zeros(self._n)
        for leaf in self._leaves:
            (M_leaf, h_leaf) = leaf.pull(q, qdot)
            self._M += M_leaf
            h_int += np.dot(M_leaf, h_leaf)
        self._h = np.dot(np.linalg.pinv(self._M), h_int)
        if self._damper:
            (alpha, beta) = self._damper.damp(q, qdot, self._h)
            self._d = alpha * qdot - beta * qdot

    def setRHS(self):
        self._rhs = -self._h + self._d

    def augment(self):
        self._rhs_aug[0] = self._qdot[0]
        self._rhs_aug[1] = self._qdot[1]
        self._rhs_aug[2] = self._rhs[0]
        self._rhs_aug[3] = self._rhs[1]

    def contDynamics(self, z, t):
        self.update(z[0:self._n], z[self._n:2*self._n])
        self.setRHS()
        self.augment()
        zdot = self._rhs_aug
        return zdot

    def computePath(self, z0, dt, T):
        t = np.arange(0.0, T, step=dt)
        sol, info = odeint(self.contDynamics, z0, t, full_output=True)
        return sol
