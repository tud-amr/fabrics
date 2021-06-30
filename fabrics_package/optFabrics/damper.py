import casadi as ca
import numpy as np
from optFabrics.functions import generateEnergizer, generateLagrangian
from optFabrics.diffMap import DiffMap
from optFabrics.execEnergy import ExecutionEnergy

class Damper(object):

    def __init__(self, beta_fun, eta_fun, le, lex, x, xdot):
        self.beta_fun = beta_fun
        self.eta_fun = eta_fun
        self.lex_fun = ca.Function("Lex", [x, xdot], [lex])
        n = x.size(1)
        self.ale_fun = generateEnergizer(le, x, xdot, "le", n)
        self.alex_fun = generateEnergizer(lex, x, xdot, "lex", n)

    def damp(self, x, xdot, h):
        h_zero = np.zeros(h.size(1))
        lex = self.lex_fun(x, xdot)
        alex0 = self.alex_fun(x, xdot, h_zero)
        alexpsi = self.alex_fun(x, xdot, h)
        eta = self.eta_fun(lex, 0.50 * lex)
        ale = self.ale_fun(x, xdot, h_zero)
        alex = eta * alex0 + (1 - eta) * alexpsi
        beta = self.beta_fun(x, xdot, ale, alex)
        return (alex, beta)

class RootDamper(object):
    def __init__(self, q, qdot, beta_fun, eta_fun, exEn):
        self.beta_fun = beta_fun
        self.eta_fun = eta_fun
        self._exEn = exEn

    def alpha(self, f, fe, M, q, qdot):
        a1 = np.dot(qdot, np.dot(M, qdot))
        a2 = np.dot(qdot, f - fe)
        return -a2/(a1 + 1e-6)

    def damp(self, f_geometry, f_forcing, fe_geometry, fe_forcing, M_geometry, M_forcing, q, qdot, x):
        ale0 = self.alpha(f_geometry, fe_forcing, M_forcing, q, qdot)
        alex0 = self._exEn.alpha(q, qdot, f_geometry)
        alexpsi = self._exEn.alpha(q, qdot, f_geometry + f_forcing)
        lex = self._exEn.energy(q, qdot)
        eta = self.eta_fun(lex)
        alex = float(eta * alex0 + (1 - eta) * alexpsi)
        beta = float(self.beta_fun(x, ale0, alex))
        return (alex, beta)

class ConstantRootDamper(object):
    def __init__(self, beta):
        self._beta = beta

    def damp(self, f_geometry, f_forcing, fe_geometry, fe_forcing, M_geometry, M_forcing, q, qdot, x):
        return (0.0, self._beta)

def createRootDamper(q, qdot, m, diffMap_ex, x_ex, xdot_ex, a_eta=0.5, a_beta=0.5, a_shift=0.5, r=1.5, b=np.array([0.03, 6.5])):
    x_forcing = ca.SX.sym("x_forcing", m)
    ale = ca.SX.sym("ale", 1)
    alex = ca.SX.sym("alex", 1)
    elex = ca.SX.sym('elex', 1)
    beta_switch = 0.5 * (ca.tanh(-a_beta * (ca.norm_2(x_forcing) - r)) + 1)
    beta = beta_switch * b[1] + b[0] + ca.fmax(0.0, alex - ale)
    eta = 0.5 * (ca.tanh(-a_eta*(elex) - a_shift) + 1)
    lex = ca.dot(xdot_ex, xdot_ex)
    exEn = ExecutionEnergy("exEn", diffMap_ex, lex)
    beta_fun = ca.Function("beta", [x_forcing, ale, alex], [beta])
    eta_fun = ca.Function("eta", [elex], [eta])
    damper = RootDamper(q, qdot, beta_fun, eta_fun, exEn)
    return damper

def createConstantRootDamper(beta):
    return ConstantRootDamper(beta)

def createDamper(x, xdot, le, a_eta=0.5, a_beta=0.5, a_shift=0.5, r=1.5, b=np.array([0.03, 6.5])):
    ale = ca.SX.sym("ale", 1)
    alex = ca.SX.sym("alex", 1)
    elex = ca.SX.sym('elex', 1)
    elex_d = ca.SX.sym('elex_d', 1)
    beta_switch = 0.5 * (ca.tanh(-a_beta * (ca.norm_2(x) - r)) + 1)
    beta = beta_switch * b[1] + b[0] + ca.fmax(0.0, alex - ale)
    eta = 0.5 * (ca.tanh(-a_eta*(elex - elex_d) - a_shift) + 1)
    lex = ca.dot(xdot, xdot)
    beta_fun = ca.Function("beta", [x, xdot, ale, alex], [beta])
    eta_fun = ca.Function("eta", [elex, elex_d], [eta])
    damper = Damper(beta_fun, eta_fun, le, lex, x, xdot)
    return damper
