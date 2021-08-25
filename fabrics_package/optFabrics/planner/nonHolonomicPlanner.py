import casadi as ca
from copy import deepcopy
from optFabrics.planner.fabricPlanner import FabricPlanner

from optFabrics.diffGeometry.diffMap import DifferentialMap
from optFabrics.diffGeometry.energy import Lagrangian
from optFabrics.diffGeometry.geometry import Geometry
from optFabrics.diffGeometry.energized_geometry import WeightedGeometry
from optFabrics.diffGeometry.speedControl import Damper
from optFabrics.helper_functions import joinVariables


class NonHolonomicPlanner(FabricPlanner):
    def __init__(self, geo: Geometry, lag: Lagrangian, J_nh: ca.SX, qdot: ca.SX, f_extra: ca.SX):
        assert isinstance(J_nh, ca.SX)
        self._J_nh = J_nh
        super().__init__(geo, lag)
        self._qdot = qdot
        self._f_extra = f_extra

    def concretize(self):
        self._eg.concretize()
        MJ = ca.mtimes(self._eg._M, self._J_nh)
        xddot = ca.mtimes(ca.pinv(MJ), -self._eg.f())
        if self._executionEnergy:
            MJ = ca.mtimes(self._eg_ex._M, self._J_nh)
            xddot = -ca.mtimes(ca.pinv(MJ), self._eg_ex.f() + ca.mtimes(ca.pinv(self._eg_ex._M), self._eg_ex._alpha * self._eg_ex.xdot()))
        if self._forcing:
            MJ = ca.mtimes(self._eg_f._M, self._J_nh)
            xddot = ca.mtimes(ca.pinv(MJ), -self._eg_f.f())
        if self._forcing and self._executionEnergy:
            MJ = ca.mtimes(self._eg_f_ex._M, self._J_nh)
            xddot = -ca.mtimes(ca.pinv(MJ), self._eg_f_ex.f() + ca.mtimes(ca.pinv(self._eg_f_ex._M), self._eg_f_ex._alpha * self._eg_f_ex.xdot()))
        if self._speedControl:
            MJ = ca.mtimes(self._eg_f._M, self._J_nh)
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
            xddot = -ca.mtimes(ca.pinv(MJ), self._eg_f.f() + ca.mtimes(self._eg_f._M, self._f_extra))  - (a_ex + beta_subst) * self._qdot
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

