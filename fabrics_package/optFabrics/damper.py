import casadi as ca
import numpy as np
from optFabrics.functions import generateEnergizer

class Damper(object):

    def __init__(self, beta_fun, eta_fun, le, lex, q, qdot):
        self.beta_fun = beta_fun
        self.eta_fun = eta_fun
        self.le_fun = ca.Function("Le", [q, qdot], [le])
        self.lex_fun = ca.Function("Lex", [q, qdot], [lex])
        n = q.size(1)
        self.ale_fun = generateEnergizer(le, q, qdot, "le", n)
        self.alex_fun = generateEnergizer(lex, q, qdot, "le", n)

    def damp(self, q, qdot, h):
        h_zero = np.zeros(h.size(1))
        le = self.le_fun(q, qdot)
        lex = self.lex_fun(q, qdot)
        eta = self.eta_fun(le, lex)
        ale = self.ale_fun(q, qdot, h_zero)
        alex0 = self.alex_fun(q, qdot, h_zero)
        alexpsi = self.alex_fun(q, qdot, h)
        alex = eta * alex0 + (1 - eta) * alexpsi
        beta = self.beta_fun(q, qdot, ale, alex)
        return (alex, beta)

class TimeVariantDamper(object):

    def __init__(self, beta_fun, eta_fun, le, lex, q, qdot, t):
        self.beta_fun = beta_fun
        self.eta_fun = eta_fun
        self.le_fun = ca.Function("Le", [q, qdot], [le])
        self.lex_fun = ca.Function("Lex", [q, qdot], [lex])
        n = q.size(1)
        self.ale_fun = generateEnergizer(le, q, qdot, "le", n)
        self.alex_fun = generateEnergizer(lex, q, qdot, "le", n)

    def damp(self, q, qdot, h, t):
        h_zero = np.zeros(h.size(1))
        le = self.le_fun(q, qdot)
        lex = self.lex_fun(q, qdot)
        eta = self.eta_fun(le, lex)
        ale = self.ale_fun(q, qdot, h_zero)
        alex0 = self.alex_fun(q, qdot, h_zero)
        alexpsi = self.alex_fun(q, qdot, h)
        alex = eta * alex0 + (1 - eta) * alexpsi
        beta = self.beta_fun(q, qdot, t,  ale, alex)
        return (alex, beta)

def createDamper(x, xdot, x_d, a_eta=0.5, a_beta=0.5, a_shift=0.5, r=1.5, b=np.array([0.03, 6.5])):
    ale = ca.SX.sym("ale", 1)
    alex = ca.SX.sym("alex", 1)
    ele = ca.SX.sym('ele', 1)
    elex = ca.SX.sym('elex', 1)
    beta_switch = 0.5 * (ca.tanh(-a_beta * (ca.norm_2(x - x_d) - r)) + 1)
    beta = beta_switch * b[1] + b[0] + ca.fmax(0.0, alex - ale)
    eta = 0.5 * (ca.tanh(-a_eta*(ele - elex) - a_shift) + 1)
    le = 0.5 * ca.norm_2(xdot)**2
    lex = 0.25 * ca.norm_2(xdot)**2
    beta_fun = ca.Function("beta", [x, xdot, ale, alex], [beta])
    eta_fun = ca.Function("eta", [ele, elex], [eta])
    damper = Damper(beta_fun, eta_fun, le, lex, x, xdot)
    return damper

def createTimeVariantDamper(x, xdot, x_d, t, a_eta=0.5, a_beta=0.5, a_shift=0.5, r=1.5, b=np.array([0.03, 6.5])):
    ale = ca.SX.sym("ale", 1)
    alex = ca.SX.sym("alex", 1)
    ele = ca.SX.sym('ele', 1)
    elex = ca.SX.sym('elex', 1)
    beta_switch = 0.5 * (ca.tanh(-a_beta * (ca.norm_2(x - x_d) - r)) + 1)
    beta = beta_switch * b[1] + b[0] + ca.fmax(0.0, alex - ale)
    eta = 0.5 * (ca.tanh(-a_eta*(ele - elex) - a_shift) + 1)
    le = 0.5 * ca.norm_2(xdot)**2
    lex = 0.25 * ca.norm_2(xdot)**2
    beta_fun = ca.Function("beta", [x, xdot, t, ale, alex], [beta])
    eta_fun = ca.Function("eta", [ele, elex], [eta])
    damper = TimeVariantDamper(beta_fun, eta_fun, le, lex, x, xdot, t)
    return damper

