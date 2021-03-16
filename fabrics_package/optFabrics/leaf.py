import casadi as ca
import numpy as np

from optFabrics.functions import createMapping, generateLagrangian, createTimeVariantMapping
from optFabrics.damper import createDamper

class DiffMap(object):
    def __init__(self, name, phi, q, qdot, x, xdot):
        (self.phi_fun, self.J_fun, self.Jdot_fun) = createMapping(phi, name, q, qdot)
        self._name = name
        self._x, self._xdot = x, xdot

    def variables(self):
        return (self._x, self._xdot)

    def forwardMap(self, q, qdot):
        x = np.array(self.phi_fun(q))[:, 0]
        J = self.J_fun(q)
        Jdot = self.Jdot_fun(q, qdot)
        Jt = np.transpose(J)
        xdot = np.dot(J, qdot)
        return (x, xdot, J, Jt, Jdot)

class TimeVariantDiffMap(DiffMap):
    def __init__(self, name, phi, q, qdot, x, xdot, t):
        (self.phi_fun, self.J_fun, self.Jdot_fun) = createTimeVariantMapping(phi, name, q,qdot, t)
        self._name = name
        self._x, self._xdot = x, xdot

    def forwardMap(self, q, qdot, t):
        x = np.array(self.phi_fun(q, t))[:, 0]
        J = self.J_fun(q, t)
        Jdot = self.Jdot_fun(q, qdot, t)
        Jt = np.transpose(J)
        xdot = np.dot(J, qdot)
        return (x, xdot, J, Jt, Jdot)

class Leaf(object):
    def __init__(self, name, diffMap, M):
        self._diffMap = diffMap
        self._x, self._xdot = diffMap.variables()
        self.M_fun = ca.Function("M_" + name, [self._x, self._xdot], [M])
        # placeholder for h_fun
        n = self._x.size(1)
        self.h_fun = ca.Function("h_" + name, [self._x, self._xdot], [np.zeros(n)])

    def pull(self, q, qdot):
        x, xdot, J, Jt, Jdot = self._diffMap.forwardMap(q, qdot)
        M = self.M_fun(x, xdot)
        M_pulled = np.dot(Jt, np.dot(M, J))
        h = self.h_fun(x, xdot)
        h_int = np.dot(Jt, np.dot(M, (h - np.dot(Jdot, qdot))))[:, 0]
        h_pulled = np.dot(np.linalg.pinv(M_pulled), h_int)
        return (M_pulled, h_pulled)

class GeometryLeaf(Leaf):
    def __init__(self, name, diffMap, M, h):
        super(GeometryLeaf, self).__init__(name, diffMap, M)
        self.h_fun = ca.Function("h_" + name, [self._x, self._xdot], [h])

class ForcingLeaf(Leaf):
    def __init__(self, name, diffMap, le, psi):
        x, xdot = diffMap.variables()
        (M, f) = generateLagrangian(le, x, xdot, "le")
        super(ForcingLeaf, self).__init__(name, diffMap, M)
        h = ca.mtimes(M, ca.gradient(psi, self._x))
        self.h_fun = ca.Function("h_" + name, [self._x, self._xdot], [h])

class DampedLeaf(ForcingLeaf):
    def __init__(self, name, diffMap, le, psi, damper):
        super(DampedLeaf, self).__init__(name, diffMap, le, psi)
        self._damper = damper

    def pull(self, q, qdot):
        x, xdot, J, Jt, Jdot = self._diffMap.forwardMap(q, qdot)
        M = self.M_fun(x, xdot)
        M_pulled = np.dot(Jt, np.dot(M, J))
        h = self.h_fun(x, xdot)
        (alpha, beta) = self._damper.damp(x, xdot, h)
        d = alpha * xdot - beta * xdot
        h -= d
        h_int = np.dot(Jt, np.dot(M, (h - np.dot(Jdot, qdot))))[:, 0]
        h_pulled = np.dot(np.linalg.pinv(M_pulled), h_int)
        return (M_pulled, h_pulled)

class TimeVariantLeaf(DampedLeaf):
    def __init__(self, name, timeVariantdiffMap, le, psi, damper):
        super(TimeVariantLeaf, self).__init__(name, diffMap, le, psi, damper)

    def pull(self, q, qdot, t):
        x, xdot, J, Jt, Jdot = self._diffMap.forwardMap(q, qdot, t)
        M = self.M_fun(x, xdot)
        M_pulled = np.dot(Jt, np.dot(M, J))
        h = self.h_fun(x, xdot)
        (alpha, beta) = self._damper.damp(x, xdot, h)
        d = alpha * xdot - beta * xdot
        h -= d
        h_int = np.dot(Jt, np.dot(M, (h - np.dot(Jdot, qdot))))[:, 0]
        h_pulled = np.dot(np.linalg.pinv(M_pulled), h_int)
        return (M_pulled, h_pulled)

def createAttractor(q, qdot, x, xdot, x_d, fk,  k=5.0, a_psi=10.0, a_m = 0.75, m=np.array([0.3, 2.0])):
    n = x.size(1)
    phi = fk - x_d
    dm = DiffMap("attractor", phi, q, qdot, x, xdot)
    psi = k * (ca.norm_2(x) + 1/a_psi * ca.log(1 + ca.exp(-2*a_psi * ca.norm_2(x))))
    M = ((m[1] - m[0]) * ca.exp(-(a_m * ca.norm_2(x))**2) + m[0]) * np.identity(n)
    le = ca.dot(xdot, ca.mtimes(M, xdot))
    damper = createDamper(x, xdot, le)
    lforcing = DampedLeaf("attractor", dm, le, psi, damper)
    return lforcing

def createQuadraticAttractor(q, qdot, x, xdot, x_d, fk,  k=5.0, a_psi=10.0, a_m = 0.75, m=np.array([0.3, 2.0])):
    n = x.size(1)
    phi = fk - x_d
    dm = DiffMap("attractor", phi, q, qdot, x, xdot)
    psi = ca.norm_2(x)**2
    #M = ((m[1] - m[0]) * ca.exp(-(a_m * ca.norm_2(x))**2) + m[0]) * np.identity(n)
    M = np.identity(n)
    le = ca.dot(xdot, ca.mtimes(M, xdot))
    damper = createDamper(x, xdot, le)
    lforcing = DampedLeaf("forcing", dm, le, psi, damper)
    return lforcing

def createExponentialAttractor(q, qdot, x, xdot, x_d, fk,  k=5.0, a_psi=10.0, a_m = 0.75, m=np.array([0.3, 2.0])):
    n = x.size(1)
    phi = fk - x_d
    dm = DiffMap("attractor", phi, q, qdot, x, xdot)
    psi = ca.exp(0.2 * ca.norm_2(x)**2)
    #M = ((m[1] - m[0]) * ca.exp(-(a_m * ca.norm_2(x))**2) + m[0]) * np.identity(n)
    M = np.identity(n)
    le = ca.dot(xdot, ca.mtimes(M, xdot))
    damper = createDamper(x, xdot, le)
    lforcing = DampedLeaf("forcing", dm, le, psi, damper)
    return lforcing

def createCollisionAvoidance(q, qdot, fk, x_obst, r_obst, lam=0.25, a=np.array([0.4, 0.2, 20.0, 5.0])):
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym('xdot', 1)
    phi = ca.norm_2(fk - x_obst) / r_obst - 1
    dm = DiffMap("attractor", phi, q, qdot, x, xdot)
    psi_col = a[0] / (x**2) + a[1] * ca.log(ca.exp(-a[2] * (x - a[3])) + 1)
    s_col = 0.5 * (ca.tanh(-10 * xdot) + 1)
    h = xdot ** 2 * lam * ca.gradient(psi_col, x)
    le = 0.5 * s_col * xdot**2 * lam/x
    (M, _) = generateLagrangian(le, x, xdot, 'col')
    lcol = GeometryLeaf("col_avo", dm, M, h)
    return lcol

def createJointLimits(q, qdot, upper_lim, lower_lim, a=np.array([0.4, 0.2, 20.0, 5.0]), lam=0.25):
    n = q.size(1)
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym('xdot', 1)
    psi_lim = a[0] / (x**2) + a[1] * ca.log(ca.exp(-a[2] * (x - a[3])) + 1)
    s_lim = 0.5 * (ca.tanh(-10 * xdot) + 1)
    h = xdot ** 2 * s_lim * lam * ca.gradient(psi_lim, x)
    le = 0.5 * xdot**2 * lam/x
    (M, f) = generateLagrangian(le, x, xdot, 'lim')
    leaves = []
    for i in range(n):
        q_min = lower_lim[i]
        q_max = upper_lim[i]
        phi_min = q[i] - q_min
        phi_max = q_max - q[i]
        dm_min = DiffMap("limitmin_" + str(i), phi_min, q, qdot, x, xdot)
        dm_max = DiffMap("limitmax_" + str(i), phi_max, q, qdot, x, xdot)
        lqmin = GeometryLeaf("qmin_" + str(i), dm_min, M, h)
        lqmax = GeometryLeaf("qmax_" + str(i), dm_max, M, h)
        leaves.append(lqmin)
        leaves.append(lqmax)
    return leaves
