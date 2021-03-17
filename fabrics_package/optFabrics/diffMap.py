import casadi as ca
import numpy as np

from optFabrics.functions import createMapping, createTimeVariantMapping

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
