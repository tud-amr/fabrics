import casadi as ca
import numpy as np
from optFabrics.functions import generateEnergizer

class Damper(object):

    def __init__(self, beta_fun, eta_fun, le, lex, q, qdot):
        self.beta_fun = beta_fun
        self.eta_fun = eta_fun
        self.le_fun = ca.Function("Le", [q, qdot], [le])
        self.lex_fun = ca.Function("Lex", [q, qdot], [lex])
        self.ale_fun = generateEnergizer(le, q, qdot, "le", 2)
        self.alex_fun = generateEnergizer(lex, q, qdot, "le", 2)

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
