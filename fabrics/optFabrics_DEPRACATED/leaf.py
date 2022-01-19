import casadi as ca
import numpy as np
import math

from optFabrics.functions import generateLagrangian, createTimeVariantMapping, generateHamiltonian
from optFabrics.damper import createDamper
from optFabrics.diffMap import DiffMap


class Leaf(object):
    def __init__(self, name, diffMap, le):
        self._diffMap = diffMap
        self._x, self._xdot = diffMap.variables()
        (Me, fe) = generateLagrangian(le, self._x, self._xdot, name)
        self.he_fun = generateHamiltonian(le, self._x, self._xdot, name)
        self.le_fun = ca.Function("le_" + name, [self._x, self._xdot], [le])
        self._Me = Me
        self.M_fun = ca.Function("M_" + name, [self._x, self._xdot], [Me])
        self.fe_fun = ca.Function("fe_" + name, [self._x, self._xdot], [fe])
        # placeholder
        f = np.zeros(self._x.size(1))
        self.f_fun = ca.Function("f_" + name, [self._x, self._xdot], [f])
        self._name = name

    def pull(self, q, qdot, t):
        x, xdot, J, Jt, Jdot = self._diffMap.forwardMap(q, qdot, t)
        M = self.M_fun(x, xdot)
        f = self.f_fun(x, xdot)
        fe = self.fe_fun(x, xdot)
        if np.linalg.norm(f) > 1000 :
            print("name : ", self.name())
            print('f : ', f)
            print("M : ", M)
            print('fe : ', fe)
            print('x : ', x)
            print('xdot : ', xdot)
        """
        if self.name() == "col_avo":
            print("name : ", self.name())
            print('x : ', x)
            print('xdot : ', xdot)
            print("f : ", f)
        """
        M_pulled = np.dot(Jt, np.dot(M, J))
        f_pulled = np.dot(Jt, f + np.dot(M, np.dot(Jdot, qdot)))[:, 0] # the minus ...
        fe_pulled = np.dot(Jt, fe + np.dot(M, np.dot(Jdot, qdot)))[:, 0] # the minus ...
        le = self.le_fun(x, xdot)
        he = self.he_fun(x, xdot)
        if np.linalg.norm(fe_pulled) < 1e-10:
            fe_pulled = np.zeros(len(fe_pulled))
        if np.linalg.norm(f_pulled) < 1e-10:
            f_pulled = np.zeros(len(f_pulled))
        return (M_pulled, f_pulled, fe_pulled, he)

    def name(self):
        return self._name

class GeometryLeaf(Leaf):
    def __init__(self, name, diffMap, le, h):
        super(GeometryLeaf, self).__init__(name, diffMap, le)
        f = ca.mtimes(self._Me, h)
        self.h_fun = ca.Function("h_" + name, [self._x, self._xdot], [h])
        self.f_fun = ca.Function("f_" + name, [self._x, self._xdot], [f])

class ForcingLeaf(Leaf):
    def __init__(self, name, diffMap, le, psi):
        x, xdot = diffMap.variables()
        super(ForcingLeaf, self).__init__(name, diffMap, le)
        f = ca.mtimes(self._Me, ca.gradient(psi, self._x))
        self.f_fun = ca.Function("f_" + name, [self._x, self._xdot], [f])
        self.he_fun = ca.Function("he_" + name, [self._x, self._xdot], [psi])

class DampedLeaf(ForcingLeaf):
    def __init__(self, name, diffMap, le, psi, damper):
        super(DampedLeaf, self).__init__(name, diffMap, le, psi)
        self._damper = damper

    def pull(self, q, qdot, t):
        x, xdot, J, Jt, Jdot = self._diffMap.forwardMap(q, qdot, t)
        M = self.M_fun(x, xdot)
        M_pulled = np.dot(Jt, np.dot(M, J))
        h = self.h_fun(x, xdot)
        (alpha, beta) = self._damper.damp(x, xdot, h)
        d = alpha * xdot - beta * xdot
        h -= d
        h_int = np.dot(Jt, np.dot(M, (h - np.dot(Jdot, qdot))))[:, 0]
        h_pulled = np.dot(np.linalg.pinv(M_pulled), h_int)
        le = self.le_fun(x, xdot)
        return (M_pulled, h_pulled, le)

class TimeVariantLeaf(DampedLeaf):
    def __init__(self, name, timeVariantdiffMap, le, psi, damper):
        super(TimeVariantLeaf, self).__init__(name, diffMap, le, psi, damper)

    def pull(self, q, qdot, t):
        x, xdot, J, Jt, Jdot = self._diffMap.forwardMap(q, qdot, t)
        M = self.M_fun(x, xdot)
        M_pulled = np.dot(Jt, np.dot(M, J))
        h = self.h_fun(x, xdot)
        (alpha, beta) = self._damper.damp(x, xdot, h)
        d = alpha * xdot - beta * xdot
        h -= d
        h_int = np.dot(Jt, np.dot(M, (h - np.dot(Jdot, qdot))))[:, 0]
        h_pulled = np.dot(np.linalg.pinv(M_pulled), h_int)
        return (M_pulled, h_pulled)

class DynamicLeaf(Leaf):
    def __init__(self, name, diffMap, le, psi, xd_ca, xd_t, t_ca):
        super(DynamicLeaf, self).__init__(name, diffMap, le)
        f = ca.mtimes(self._Me, ca.gradient(psi, self._x))
        """
        print("Me : ", self._Me)
        print("f : ", f)
        print("psi : ", psi)
        """
        self.f_fun = ca.Function("f_" + name, [self._x, self._xdot, xd_ca], [f])
        xd_dot_t = ca.jacobian(xd_t, t_ca)
        xd_ddot_t = ca.jacobian(xd_dot_t, t_ca)
        self.xd_fun = ca.Function("xd_" + name, [t_ca], [xd_t])
        self.xd_dot_fun = ca.Function("xd_dot_" + name, [t_ca], [xd_dot_t])
        self.xd_ddot_fun = ca.Function("xd_ddot_" + name, [t_ca], [xd_ddot_t])

    def pull(self, q, qdot, t):
        x, xdot, J, Jt, Jdot = self._diffMap.forwardMap(q, qdot, t)
        self._Jt = Jt
        xd = np.array(self.xd_fun(t))[:, 0]
        xd_dot = np.array(self.xd_dot_fun(t))[:, 0]
        print("x : ", x)
        print("xdot : ", xdot)
        print("x_d : ", xd)
        print("xd_dot : ", xd_dot)
        self._xd_dot = xd_dot
        xd_ddot = np.array(self.xd_ddot_fun(t))[:, 0]
        print("xd_ddot : ", xd_ddot)
        print("M xddot_d : ", xd_ddot)
        M = self.M_fun(x, xdot)
        fe = np.array(self.fe_fun(x, xdot))[:, 0]
        f_psi = np.array(self.f_fun(x, xdot, xd))[:, 0]
        f_d = np.dot(M, xd_ddot)
        # this damping should not be necessary
        #b_d = self._beta * (xdot - xd_dot)
        #f = f_psi - f_d + b_d
        f = f_psi - f_d
        print("f_psi in leaf : ", f_psi)
        print("f in leaf : ", f)
        print("fd in leaf : ", f_d)
        print("M in leaf : ", M)
        M_pulled = np.dot(Jt, np.dot(M, J))
        f_pulled = np.dot(Jt, f + np.dot(M, np.dot(Jdot, qdot)))
        fe_pulled = np.dot(Jt, fe + np.dot(M, np.dot(Jdot, qdot)))
        print("f_pulled : ", f_pulled)
        print("M_pulled : ", M_pulled)
        he = self.he_fun(x, xdot)
        return (M_pulled, f_pulled, fe_pulled, he)

    def bxddot(self, beta):
        val = -np.dot(self._Jt, beta * self._xd_dot)
        return val

class SplineLeaf(Leaf):
    def __init__(self, name, diffMap, le, psi, xd, spline, T, dt, beta=1.0):
        super(SplineLeaf, self).__init__(name, diffMap, le)
        f = ca.mtimes(self._Me, ca.gradient(psi, self._x))
        ctrlDim = len(spline.ctrlpts[0])
        self._spline = spline
        self._beta = beta
        self._scaling = 1/T
        self._dt = dt
        nbEvals = 10000
        points = np.array(spline.evaluate_list([1/nbEvals * i for i in range(nbEvals)]))
        spline_length = sum([np.linalg.norm(points[i] - points[i-1]) for i in range(1, nbEvals)])
        self._vn = 1.5 * spline_length*self._scaling
        self._vn = 0.3
        #print(self._vn)
        self.f_fun = ca.Function("f_" + name, [self._x, xd], [f])

    def pull(self, q, qdot, t):
        x, xdot, J, Jt, Jdot = self._diffMap.forwardMap(q, qdot, t)
        self._Jt = Jt
        M = self.M_fun(x, xdot)
        fe = self.fe_fun(x, xdot)
        t_ideal = t * self._scaling
        t_close = project(x, self._spline)
        t_ref = min((t_ideal + t_close)/2, 1)
        xds = self._spline.derivatives(t_ref, order=2)
        xd = np.array(xds[0])
        v_s = max(0.0, self._vn * (-1 *((t_ref/0.5) - 1)**2 + 1))
        a_s = max(0.0, self._vn * (-2 *((t_ref/0.5) - 1)))
        #print("x : ", x, " , x_d : ", xd)
        xd_dot = v_s * np.array(xds[1])
        xd_ddot = a_s * np.array(xds[2])
        self._xd_dot = xd_dot
        f_psi = self.f_fun(x, xd)
        f_d = np.dot(M, xd_ddot)
        #b_d = self._beta * (xdot - xd_dot)
        f = f_psi# - f_d
        #f = -f_d + f_psi + beta * (xdot - xd_dot)
        M_pulled = np.dot(Jt, np.dot(M, J))
        f_pulled = np.dot(Jt, f + np.dot(M, np.dot(Jdot, qdot)))[:, 0] # the minus ...
        fe_pulled = np.dot(Jt, fe + np.dot(M, np.dot(Jdot, qdot)))[:, 0] # the minus ...
        he = self.he_fun(x, xdot)
        return (M_pulled, f_pulled, fe_pulled, he)

    def bxddot(self, beta):
        val = -np.dot(self._Jt, beta * self._xd_dot)
        return val

class DiffDriveSplineLeaf(Leaf):
    def __init__(self, name, diffMap, le, psi, xd, spline, T, beta=1.0):
        super(DiffDriveSplineLeaf, self).__init__(name, diffMap, le)
        f = ca.mtimes(self._Me, ca.gradient(psi, self._x))
        self._spline = spline
        self._beta = beta
        self._scaling = 1/T
        nbEvals = 10000
        points = np.array(spline.evaluate_list([1/nbEvals * i for i in range(nbEvals)]))
        spline_length = sum([np.linalg.norm(points[i] - points[i-1]) for i in range(1, nbEvals)])
        self._vn = 1.5 * spline_length*self._scaling
        self._vn = 0.3
        #print(self._vn)
        self.f_fun = ca.Function("f_" + name, [self._x, xd], [f])

    def pull(self, q, qdot, t):
        x, xdot, J, Jt, Jdot = self._diffMap.forwardMap(q, qdot, t)
        self._Jt = Jt
        M = self.M_fun(x, xdot)
        fe = self.fe_fun(x, xdot)
        t_ref = min(self._scaling * t, 1)
        xds = self._spline.derivatives(t_ref, order=4)
        thetad = math.atan2(xds[1][1], xds[1][0])
        thetad_dot = math.atan2(xds[2][1], xds[2][0])
        thetad_ddot = math.atan2(xds[3][1], xds[3][0])
        xd = np.array([xds[0][0], xds[0][1], thetad])
        v_s = max(0.0, self._vn * (-1 *((t_ref/0.5) - 1)**2 + 1))
        a_s = max(0.0, self._vn * (-2 *((t_ref/0.5) - 1)))
        xd_dot = v_s * np.array([xds[1][0], xds[1][1], thetad_dot])
        xd_ddot = a_s * np.array([xds[2][0], xds[2][1], thetad_ddot])
        self._xd_dot = xd_dot
        f_psi = self.f_fun(x, xd)
        f_d = np.dot(M, xd_ddot)
        #b_d = self._beta * (xdot - xd_dot)
        f = f_psi# - f_d
        #f = -f_d + f_psi + beta * (xdot - xd_dot)
        M_pulled = np.dot(Jt, np.dot(M, J))
        f_pulled = np.dot(Jt, f + np.dot(M, np.dot(Jdot, qdot)))[:, 0] # the minus ...
        fe_pulled = np.dot(Jt, fe + np.dot(M, np.dot(Jdot, qdot)))[:, 0] # the minus ...
        he = self.he_fun(x, xdot)
        return (M_pulled, f_pulled, fe_pulled, he)

    def bxddot(self, beta):
        val = -np.dot(self._Jt, beta * self._xd_dot)
        return val

