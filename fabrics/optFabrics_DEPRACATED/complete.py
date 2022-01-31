import numpy as np
from scipy.integrate import odeint
import casadi as ca

from damper import Damper
from leaf import Leaf
from rootGeometry import RootGeometry
from functions import createMapping, generateLagrangian, generateEnergizer
from plottingGeometries import plotTraj, animate, plotMultipleTraj, plot, plotMulti

def createMapping(phi, name):
    # differential map and jacobian phi : Q -> X
    phi_fun = ca.Function("phi_" + name, [q], [phi])
    J = ca.jacobian(phi, q)
    Jdot = ca.jacobian(ca.mtimes(J, qdot), q)
    J_fun = ca.Function("J_" + name, [q], [J])
    Jdot_fun = ca.Function("Jdot_" + name, [q, qdot], [Jdot])
    return (phi_fun, J_fun, Jdot_fun)

def generateLagrangian(L, q, qdot, name):
    dL_dq = ca.gradient(L, q)
    dL_dqdot = ca.gradient(L, qdot)
    d2L_dq2 = ca.jacobian(dL_dq, q)
    d2L_dqdqdot = ca.jacobian(dL_dq, qdot)
    d2L_dqdot2 = ca.jacobian(dL_dqdot, qdot)

    M = d2L_dqdot2
    F = d2L_dqdqdot
    f_e = -dL_dq
    f = ca.mtimes(ca.transpose(F), qdot) + f_e
    return (M, f)

def generateEnergizer(L, q, qdot, name, n):
    (Me, fe) = generateLagrangian(L, q, qdot, name)
    h = ca.SX.sym("h", n)
    a1 = ca.dot(qdot, ca.mtimes(Me, qdot))
    a2 = ca.dot(qdot, ca.mtimes(Me, h) - fe)
    a = a2/a1
    a_fun = ca.Function('a_' + name, [q, qdot, h], [a])
    return a_fun

q = ca.SX.sym("q", 2)
qdot = ca.SX.sym("qdot", 2)

class Leaf(object):
    def __init__(self, name, phi, M, h, x, xdot):
        self.h_fun = ca.Function("h_" + name, [x, xdot], [h])
        self.M_fun = ca.Function("M_" + name, [x, xdot], [M])
        (self.phi_fun, self.J_fun, self.Jdot_fun) = createMapping(phi, name)

    def pull(self, q, qdot):
        J = self.J_fun(q)
        Jdot = self.Jdot_fun(q, qdot)
        Jt = np.transpose(J)
        x = np.array(self.phi_fun(q))[:, 0]
        xdot = np.dot(J, qdot)
        M = self.M_fun(x, xdot)
        M_pulled = np.dot(Jt, np.dot(M, J))
        h = self.h_fun(x, xdot)
        #h_int = np.dot(Jt, np.dot(M, (h + np.dot(Jdot, qdot))))[:, 0] #according to 3.1.1
        h_int = np.dot(Jt, np.dot(M, (h - np.dot(Jdot, qdot))))[:, 0] #according to 6.2
        h_pulled = np.dot(np.linalg.pinv(M_pulled), h_int)
        return (M_pulled, h_pulled)

class Damper(object):

    def __init__(self, forcingLeave, beta_fun, eta_fun, le, lex, q, qdot):
        self._fl = forcingLeave
        self.beta_fun = beta_fun
        self.eta_fun = eta_fun
        self.le_fun = ca.Function("Le", [q, qdot], [le])
        self.lex_fun = ca.Function("Lex", [q, qdot], [lex])
        self.ale_fun = generateEnergizer(le, q, qdot, "le", 2)
        self.alex_fun = generateEnergizer(lex, q, qdot, "le", 2)

    def damp(self, q, qdot, h):
        (M_pulled, h_pulled) = self._fl.pull(q, qdot)
        le = self.le_fun(q, qdot)
        lex = self.lex_fun(q, qdot)
        eta = self.eta_fun(le, lex)
        ale = self.ale_fun(q, qdot, h - h_pulled)
        alex0 = self.alex_fun(q, qdot, h - h_pulled)
        alexpsi = self.alex_fun(q, qdot, h)
        alex = eta * alex0 + (1 - eta) * alexpsi
        beta = self.beta_fun(q, qdot, ale, alex)
        return (alex, beta)

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

def setBoundaryLeaves():
    # Define boundary leaves
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym('xdot', 1)
    lam = 0.25
    a = np.array([0.4, 0.2, 20.0, 5.0])
    psi_lim = a[0] / (x**2) + a[1] * ca.log(ca.exp(-a[2] * (x - a[3])) + 1)
    h_lim = xdot ** 2 * lam * ca.gradient(psi_lim, x)
    L_lim = 0.5 * xdot**2 * lam/x
    (M_lim, f_lim) = generateLagrangian(L_lim, x, xdot, 'lim')
    # Define leaf
    q_min = -4.0
    phi_lim = q[0] - q_min
    lxlow = Leaf("xlim_low", phi_lim, M_lim, h_lim, x, xdot)
    # ---
    # Define leaf
    q_min = -4.0
    phi_lim = q[1] - q_min
    lylow = Leaf("ylim_low", phi_lim, M_lim, h_lim, x, xdot)
    # ---
    # Define leaf
    q_max = 4.0
    phi_lim = q_max - q[1]
    lyup = Leaf("xlim_up", phi_lim, M_lim, h_lim, x, xdot)
    # ---
    # Define leaf
    q_max = 4.0
    phi_lim = q_max - q[0]
    lxup = Leaf("ylim_up", phi_lim, M_lim, h_lim, x, xdot)
    # ---
    return (lxup, lxlow, lyup, lylow)

def createCollisionAvoidance():
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym('xdot', 1)
    lam = 0.25
    a = np.array([0.4, 0.2, 20.0, 5.0])
    psi_col = a[0] / (x**2) + a[1] * ca.log(ca.exp(-a[2] * (x - a[3])) + 1)
    h_col = xdot ** 2 * lam * ca.gradient(psi_col, x)
    L_col = 0.5 * xdot**2 * lam/x
    (M_col, f_col) = generateLagrangian(L_col, x, xdot, 'col')
    print(M_col)
    q_obst = np.array([0.0, 0.0])
    r_obst = 1.0
    phi_col = ca.norm_2(q - q_obst) / r_obst - 1
    lcol = Leaf("col_avo", phi_col, M_col, h_col, x, xdot)
    return lcol

def forcingLeaf():
    x = ca.SX.sym("x", 2)
    xdot = ca.SX.sym("xdot", 2)
    q_d = np.array([-2.5, -3.75])
    k = 5.0
    alpha_psi = 10.0
    alpha_m = 0.75
    m = np.array([0.3, 2.0])

    phi = q - q_d
    psi = k * (ca.norm_2(x) + 1/alpha_psi * ca.log(1 + ca.exp(-2*alpha_psi * ca.norm_2(x))))
    M_forcing = ((m[1] - m[0]) * ca.exp(-(alpha_m * ca.norm_2(x))**2) + m[0]) * np.identity(2)
    h_forcing = ca.mtimes(M_forcing, ca.gradient(psi, x))
    lforcing = Leaf("forcing", phi, M_forcing, h_forcing, x, xdot)
    return lforcing

def createDamper(forcingLeave):
    ale = ca.SX.sym("ale", 1)
    alex = ca.SX.sym("alex", 1)
    ele = ca.SX.sym('ele', 1)
    elex = ca.SX.sym('elex', 1)
    a_eta = 0.5
    a_beta = 0.5
    a_shift = 0.5
    r = 1.5
    b = np.array([0.01, 6.5])
    q_d = np.array([-2.5, -3.75])
    beta_switch = 0.5 * (ca.tanh(-a_beta * (ca.norm_2(q - q_d) - r)) + 1)
    beta = beta_switch * b[1] + b[0] + ca.fmax(0.0, alex - ale)
    eta = 0.5 * (ca.tanh(-a_eta*(ele - elex) - a_shift) + 1)
    le = 0.5 * ca.norm_2(qdot)**2
    lex = 0.25 * ca.norm_2(qdot)**2
    # Functions
    beta_fun = ca.Function("beta", [q, qdot, ale, alex], [beta])
    eta_fun = ca.Function("eta", [ele, elex], [eta])
    damper = Damper(forcingLeave, beta_fun, eta_fun, le, lex, q, qdot)
    return damper


if __name__ == "__main__":
    (lxup, lxlow, lyup, lylow) = setBoundaryLeaves()
    lforcing = forcingLeaf()
    lcol = createCollisionAvoidance()
    damper = createDamper(lforcing)
    rg = RootGeometry([], 2)
    rg_forced = RootGeometry([lforcing], 2, damper=damper)
    rg_limits = RootGeometry([lyup, lylow, lxup, lxlow], 2)
    rg_forced_limits = RootGeometry([lyup, lylow, lxup, lxlow, lforcing], 2, damper=damper)
    rg_col = RootGeometry([lcol], 2)
    rg_col_limits = RootGeometry([lcol, lyup, lylow, lxup, lxlow], 2)
    rg_col_limits_forced = RootGeometry([lcol, lyup, lylow, lxup, lxlow, lforcing], 2, damper=damper)
    geos = [rg, rg_forced, rg_limits, rg_forced_limits, rg_col_limits, rg_col_limits_forced]
    # solve
    dt = 0.01
    T = 16.0
    sols = []
    aniSols = []
    x0 = np.array([2.0, 3.0])
    x0dot_norm = 1.5
    init_angles = [((i+0.13) * np.pi)/7 for i in range(14)]
    for geo in geos:
        geoSols = []
        for i, a in enumerate(init_angles):
            print("Plotting for ", a)
            x0dot = np.array([np.cos(a), np.sin(a)]) * x0dot_norm
            z0 = np.concatenate((x0, x0dot))
            sol = geo.computePath(z0, dt, T)
            geoSols.append(sol)
            if i == 0:
                aniSols.append(sol)
        sols.append(geoSols)
    plotMulti(sols, aniSols)

