import casadi as ca
import numpy as np
from copy import deepcopy
from optFabrics.planner.fabricPlanner import FabricPlanner

from optFabrics.diffGeometry.diffMap import DifferentialMap
from optFabrics.diffGeometry.energy import Lagrangian
from optFabrics.diffGeometry.geometry import Geometry
from optFabrics.diffGeometry.energized_geometry import WeightedGeometry
from optFabrics.diffGeometry.speedControl import Damper
from optFabrics.diffGeometry.variables import eps
from optFabrics.helper_functions import joinVariables


class NonHolonomicPlanner(FabricPlanner):
    def __init__(
        self, geo: Geometry, lag: Lagrangian, J_nh: ca.SX, qdot: ca.SX, f_extra: ca.SX
    ):
        assert isinstance(J_nh, ca.SX)
        self._J_nh = J_nh
        super().__init__(geo, lag)
        self._qdot = qdot
        self._n -= 1
        self._f_extra = f_extra

    def vars(self):
        return self._vars[0], self._vars[1], self._qdot

    def concretize(self):
        self._eg.concretize()
        MJ = ca.mtimes(self._eg._M, self._J_nh)
        MJtMJ = ca.mtimes(ca.transpose(MJ), MJ) + ca.SX(np.identity(self._n)) * eps
        MJ_pinv = ca.mtimes(ca.inv(MJtMJ), ca.transpose(MJ))
        xddot = ca.mtimes(MJ_pinv, -self._eg.f())
        if self._executionEnergy:
            xddot = ca.mtimes(
                MJ_pinv,
                - self._eg.f()
                - ca.mtimes(self._eg._M, self._f_extra)
                + ca.mtimes(self._eg._M, self._eg_ex._alpha * self._eg.xdot())
            )
        if self._forcing:
            MJ = ca.mtimes(self._eg_f._M, self._J_nh)
            MJtMJ = ca.mtimes(ca.transpose(MJ), MJ) + ca.SX(np.identity(self._n)) * eps
            MJ_pinv = ca.mtimes(ca.inv(MJtMJ), ca.transpose(MJ))
            xddot = ca.mtimes(MJ_pinv, -self._eg_f.f())
        if self._forcing and self._executionEnergy:
            MJ = ca.mtimes(self._eg_f._M, self._J_nh)
            MJtMJ = ca.mtimes(ca.transpose(MJ), MJ) + ca.SX(np.identity(self._n)) * eps
            MJ_pinv = ca.mtimes(ca.inv(MJtMJ), ca.transpose(MJ))
            xddot = ca.mtimes(
                MJ_pinv,
                - self._eg_f.f()
                - ca.mtimes(self._eg_f._M, self._f_extra)
                + ca.mtimes(self._eg_f._M, self._eg_f_ex._alpha * self._eg_f.xdot())
            )
        if self._speedControl:
            MJ = ca.mtimes(self._eg_f._M, self._J_nh)
            MJtMJ = ca.mtimes(ca.transpose(MJ), MJ) + ca.SX(np.identity(self._n)) * eps
            MJ_pinv = ca.mtimes(ca.inv(MJtMJ), ca.transpose(MJ))
            if self._constantSpeedControl:
                beta_subst = self._constant_beta
                if self._executionEnergy:
                    a_ex = self._eg_f_ex._alpha
                else:
                    a_ex = 0.0
            else:
                a_ex = (
                    self._eta * self._eg._alpha + (1 - self._eta) * self._eg_f_ex._alpha
                )
                beta_subst = self._beta.substitute(-a_ex, -self._eg._alpha)
            xddot = ca.mtimes(
                MJ_pinv,
                - self._eg_f.f()
                - ca.mtimes(self._eg_f._M, self._f_extra)
                - ca.mtimes(self._eg_f._M, a_ex * self._eg_f.xdot())
            ) - beta_subst * self._qdot
            """
            xddot = (
                -ca.mtimes(
                    MJ_pinv, self._eg_f.f() + ca.mtimes(self._eg_f._M, self._f_extra)
                )
                - (a_ex + beta_subst) * self._qdot
            )
            xddot = -ca.mtimes(
                MJ_pinv, self._eg_f.f() + ca.mtimes(self._eg_f._M, self._f_extra)
            )
            """
            # xddot = -ca.mtimes(ca.pinv(MJ), self._eg_f._f - ca.mtimes(ca.pinv(self._eg_f._M), (a_ex + beta_subst) * self._eg_f.xdot()))

        """
        if self._forcing:
            MJ = ca.mtimes(self._eg_f._M, self._J_nh)
            xddot = ca.mtimes(ca.pinv(MJ), -self._eg_f._f)
        if self._forcing and self._constantSpeedControl:
            MJ = ca.mtimes(self._eg_f._M, self._J_nh)
            xddot = ca.mtimes(ca.pinv(MJ), -self._eg_f._f) - self._constant_beta * self._qdot
        """
        totalVar = deepcopy(self._vars)
        for refTraj in self._refTrajs:
            totalVar += refTraj._vars
        totalVar.append(self._qdot)
        self._funs = ca.Function("planner", totalVar, [xddot])


class DefaultNonHolonomicPlanner(NonHolonomicPlanner):
    def __init__(self, n: int, **kwargs):
        p = {'m_base': 0.5, 'debug': False}
        q = ca.SX.sym("q", n)
        qdot = ca.SX.sym("qdot", n)
        qu = ca.SX.sym('q', n-1)
        qudot = ca.SX.sym("qdot", n-1)
        l_base = 0.5 * p['m_base'] * ca.dot(qdot, qdot)
        h_base = ca.SX(np.zeros(n))
        baseGeo = Geometry(h=h_base, x=q, xdot=qdot)
        baseLag = Lagrangian(l_base, x=q, xdot=qdot)
        J_nh = ca.SX(np.zeros((n, n-1)))
        J_nh[0:2, 0] = np.array([ca.cos(q[2]), ca.sin(q[2])])
        for i in range(2, n):
            J_nh[i, i-1] = 1
        f_extra = ca.SX(np.zeros((n, 1)))
        f_extra[0:2] = qudot[0] * qudot[1] * ca.vertcat(-ca.sin(q[2]), ca.cos(q[2]))
        super().__init__(baseGeo, baseLag, J_nh, qudot, f_extra)
