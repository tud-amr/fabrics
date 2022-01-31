import casadi as ca
import numpy as np

from optFabrics.functions import generateLagrangian, createTimeVariantMapping
from optFabrics.diffMap import DiffMap, TimeVariantDiffMap

class ExecutionEnergy(object):
    def __init__(self, name, diffMap, le):
        self._diffMap = diffMap
        self._x, self._xdot = diffMap.variables()
        (Me, fe) = generateLagrangian(le, self._x, self._xdot, name)
        self._Me = Me
        self.M_fun = ca.Function("M_" + name, [self._x, self._xdot], [Me])
        self.fe_fun = ca.Function("fe_" + name, [self._x, self._xdot], [fe])
        self.lex_fun = ca.Function("le_" + name, [self._x, self._xdot], [le])

    def energy(self, q, qdot):
        t = 0.0
        x, xdot, _, _, _ = self._diffMap.forwardMap(q, qdot, t)
        lex = self.lex_fun(x, xdot)
        return lex

    def alpha(self, q, qdot, f):
        t = 0.0
        x, xdot, J, Jt, Jdot = self._diffMap.forwardMap(q, qdot, t)
        M = self.M_fun(x, xdot)
        fe = self.fe_fun(x, xdot)
        M_pulled = np.dot(Jt, np.dot(M, J))
        fe_pulled = np.dot(Jt, fe - np.dot(M, np.dot(Jdot, qdot)))[:, 0] # the minus ...
        a1 = np.dot(qdot, np.dot(M_pulled, qdot))
        a2 = np.dot(qdot, f - fe_pulled)
        alpha = -a2/(a1 + 1e-6)
        return alpha

