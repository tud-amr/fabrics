import casadi as ca
from functions import generateEnergizer

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
