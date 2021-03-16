import casadi as ca
import numpy as np
from optFabrics.functions import generateEnergizer

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
