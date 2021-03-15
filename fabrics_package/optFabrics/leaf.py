import casadi as ca
import numpy as np

from optFabrics.functions import createMapping, generateLagrangian, createTimeVariantMapping
from optFabrics.damper import createDamper, createTimeVariantDamper

class Leaf(object):
    def __init__(self, name, phi, M, h, x, xdot, q, qdot, damper=None):
        self.h_fun = ca.Function("h_" + name, [x, xdot], [h])
        self.M_fun = ca.Function("M_" + name, [x, xdot], [M])
        (self.phi_fun, self.J_fun, self.Jdot_fun) = createMapping(phi, name, q, qdot)
        self._damper = damper

    def pull(self, q, qdot):
        J = self.J_fun(q)
        Jdot = self.Jdot_fun(q, qdot)
        x = np.array(self.phi_fun(q))[:, 0]
        xdot = np.dot(J, qdot)
        Jt = np.transpose(J)
        M = self.M_fun(x, xdot)
        M_pulled = np.dot(Jt, np.dot(M, J))
        h = self.h_fun(x, xdot)
        if self._damper:
            (alpha, beta) = self._damper.damp(x, xdot, h)
            d = alpha * xdot - beta * xdot
            h -= d

        #h_int = np.dot(Jt, np.dot(M, (h + np.dot(Jdot, qdot))))[:, 0] #according to 3.1.1
        h_int = np.dot(Jt, np.dot(M, (h - np.dot(Jdot, qdot))))[:, 0] #according to 6.2
        h_pulled = np.dot(np.linalg.pinv(M_pulled), h_int)
        return (M_pulled, h_pulled)

class TimeVariantLeaf(object):
    def __init__(self, name, phi, M, h, x, xdot, q, qdot, t, damper=None):
        self.h_fun = ca.Function("h_" + name, [x, xdot], [h])
        self.M_fun = ca.Function("M_" + name, [x, xdot], [M])
        (self.phi_fun, self.J_fun, self.Jdot_fun) = createTimeVariantMapping(phi, name, q, qdot, t)
        self._damper = damper

    def pull(self, q, qdot, t):
        J = self.J_fun(q, t)
        Jdot = self.Jdot_fun(q, qdot, t)
        x = np.array(self.phi_fun(q, t))[:, 0]
        xdot = np.dot(J, qdot)
        Jt = np.transpose(J)
        M = self.M_fun(x, xdot)
        M_pulled = np.dot(Jt, np.dot(M, J))
        h = self.h_fun(x, xdot)
        if self._damper:
            (alpha, beta) = self._damper.damp(x, xdot, h, t)
            d = alpha * xdot - beta * xdot
            h -= d

        #h_int = np.dot(Jt, np.dot(M, (h + np.dot(Jdot, qdot))))[:, 0] #according to 3.1.1
        h_int = np.dot(Jt, np.dot(M, (h - np.dot(Jdot, qdot))))[:, 0] #according to 6.2
        h_pulled = np.dot(np.linalg.pinv(M_pulled), h_int)
        return (M_pulled, h_pulled)

def createAttractor(q, qdot, x, xdot, x_d, fk,  k=5.0, a_psi=10.0, a_m = 0.75, m=np.array([0.3, 2.0])):
    n = x.size(1)
    phi = fk - x_d
    psi = k * (ca.norm_2(x) + 1/a_psi * ca.log(1 + ca.exp(-2*a_psi * ca.norm_2(x))))
    H = ca.hessian(psi, x)
    print("n : ", n)
    print(H)
    print("------")
    M_forcing = ((m[1] - m[0]) * ca.exp(-(a_m * ca.norm_2(x))**2) + m[0]) * np.identity(n)
    h_forcing = ca.mtimes(M_forcing, ca.gradient(psi, x))
    damper = createDamper(x, xdot, x_d)
    lforcing = Leaf("forcing", phi, M_forcing, h_forcing, x, xdot, q, qdot, damper=None)
    return lforcing

def createQuadraticAttractor(q, qdot, x, xdot, x_d, fk,  k=5.0, a_psi=10.0, a_m = 0.75, m=np.array([0.3, 2.0])):
    print("Quad : ")
    n = x.size(1)
    phi = fk - x_d
    psi = ca.norm_2(x)**2
    H = ca.hessian(psi, x)
    print("n : ", n)
    print(H)
    print("------")
    #M_forcing = ((m[1] - m[0]) * ca.exp(-(a_m * ca.norm_2(x))**2) + m[0]) * np.identity(n)
    M_forcing = np.identity(n)
    h_forcing = ca.mtimes(M_forcing, ca.gradient(psi, x))
    damper = createDamper(x, xdot, x_d)
    lforcing = Leaf("forcing", phi, M_forcing, h_forcing, x, xdot, q, qdot, damper=None)
    return lforcing

def createExponentialAttractor(q, qdot, x, xdot, x_d, fk,  k=5.0, a_psi=10.0, a_m = 0.75, m=np.array([0.3, 2.0])):
    print("Exp : ")
    n = x.size(1)
    phi = fk - x_d
    psi = ca.exp(0.2 * ca.norm_2(x)**2)
    H = ca.hessian(psi, x)
    print("n : ", n)
    print(H)
    print("------")
    #M_forcing = ((m[1] - m[0]) * ca.exp(-(a_m * ca.norm_2(x))**2) + m[0]) * np.identity(n)
    M_forcing = np.identity(n)
    h_forcing = ca.mtimes(M_forcing, ca.gradient(psi, x))
    damper = createDamper(x, xdot, x_d)
    lforcing = Leaf("forcing", phi, M_forcing, h_forcing, x, xdot, q, qdot, damper=None)
    return lforcing

def createTimeVariantAttractor(q, qdot, x, xdot, x_d, fk, t, k=5.0, a_psi=10.0, a_m = 0.75, m=np.array([0.3, 2.0])):
    n = x.size(1)
    phi = fk - x_d
    psi = k * (ca.norm_2(x) + 1/a_psi * ca.log(1 + ca.exp(-2*a_psi * ca.norm_2(x))))
    M_forcing = ((m[1] - m[0]) * ca.exp(-(a_m * ca.norm_2(x))**2) + m[0]) * np.identity(n)
    h_forcing = ca.mtimes(M_forcing, ca.gradient(psi, x))
    damper = createTimeVariantDamper(x, xdot, x_d, t)
    lforcing = TimeVariantLeaf("forcing", phi, M_forcing, h_forcing, x, xdot, q, qdot, t, damper=damper)
    return lforcing

def createCollisionAvoidance(q, qdot, fk, x_obst, r_obst, lam=0.25, a=np.array([0.4, 0.2, 20.0, 5.0])):
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym('xdot', 1)
    psi_col = a[0] / (x**2) + a[1] * ca.log(ca.exp(-a[2] * (x - a[3])) + 1)
    s_col = 0.5 * (ca.tanh(-10 * xdot) + 1)
    h_col = xdot ** 2 * lam * ca.gradient(psi_col, x)
    L_col = 0.5 * s_col * xdot**2 * lam/x
    (M_col, f_col) = generateLagrangian(L_col, x, xdot, 'col')
    phi_col = ca.norm_2(fk - x_obst) / r_obst - 1
    lcol = Leaf("col_avo", phi_col, M_col, h_col, x, xdot, q, qdot)
    return lcol

def createJointLimits(q, qdot, upper_lim, lower_lim, a=np.array([0.4, 0.2, 20.0, 5.0]), lam=0.25):
    n = q.size(1)
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym('xdot', 1)
    psi_lim = a[0] / (x**2) + a[1] * ca.log(ca.exp(-a[2] * (x - a[3])) + 1)
    s_lim = 0.5 * (ca.tanh(-10 * xdot) + 1)
    h_lim = xdot ** 2 * s_lim * lam * ca.gradient(psi_lim, x)
    L_lim = 0.5 * xdot**2 * lam/x
    (M_lim, f_lim) = generateLagrangian(L_lim, x, xdot, 'lim')
    leaves = []
    for i in range(n):
        q_min = lower_lim[i]
        q_max = upper_lim[i]
        phi_min = q[i] - q_min
        phi_max = q_max - q[i]
        lqmin = Leaf("qmin_" + str(i), phi_min, M_lim, h_lim, x, xdot, q, qdot)
        lqmax = Leaf("qmax_" + str(i), phi_max, M_lim, h_lim, x, xdot, q, qdot)
        leaves.append(lqmin)
        leaves.append(lqmax)
    return leaves
