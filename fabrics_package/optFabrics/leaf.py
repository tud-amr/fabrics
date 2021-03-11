import casadi as ca
import numpy as np

from optFabrics.functions import createMapping

class Leaf(object):
    def __init__(self, name, phi, M, h, x, xdot, q, qdot, damper=None):
        self.h_fun = ca.Function("h_" + name, [x, xdot], [h])
        self.M_fun = ca.Function("M_" + name, [x, xdot], [M])
        (self.phi_fun, self.J_fun, self.Jdot_fun) = createMapping(phi, name, q, qdot)
        self._damper = damper

    def pull(self, q, qdot):
        J = self.J_fun(q)
        Jdot = self.Jdot_fun(q, qdot)
        Jt = np.transpose(J)
        x = np.array(self.phi_fun(q))[:, 0]
        xdot = np.dot(J, qdot)
        M = self.M_fun(x, xdot)
        M_pulled = np.dot(Jt, np.dot(M, J))
        h = self.h_fun(x, xdot)
        if self._damper:
            (alpha, beta) = self._damper.damp(q, qdot, h)
            d = alpha * qdot - beta * qdot
            h -= d
        #h_int = np.dot(Jt, np.dot(M, (h + np.dot(Jdot, qdot))))[:, 0] #according to 3.1.1
        h_int = np.dot(Jt, np.dot(M, (h - np.dot(Jdot, qdot))))[:, 0] #according to 6.2
        h_pulled = np.dot(np.linalg.pinv(M_pulled), h_int)
        return (M_pulled, h_pulled)
