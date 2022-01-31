import casadi as ca
import numpy as np
from copy import deepcopy

from fabrics.diffGeometry.diffMap import DifferentialMap
from fabrics.diffGeometry.energy import Lagrangian
from fabrics.diffGeometry.geometry import Geometry
from fabrics.diffGeometry.energized_geometry import WeightedGeometry
from fabrics.diffGeometry.speedControl import Damper
from fabrics.helpers.functions import joinVariables, joinRefTrajs

from fabrics.diffGeometry.variables import eps


class FabricPlannerException(Exception):
    def __init__(self, expression, message):
        self._expression = expression
        self._message = message

    def what(self):
        return self._expression + ": " + self._message


class FabricPlanner:
    """description"""

    def __init__(self, geo: Geometry, lag: Lagrangian, debug=False):
        assert isinstance(lag, Lagrangian)
        self._eg = WeightedGeometry(g=geo, le=lag)
        self._n = lag.x().size()[0]
        self._forcing = False
        self._targetVelocity = np.zeros(self._eg.x().size()[0])
        self._executionEnergy = False
        self._speedControl = False
        self._constantSpeedControl = False
        self._vars = self._eg._vars
        self._refTrajs = []
        self._debug = debug
        self._debugVars = []

    def var(self):
        try:
            return joinVariables(self._eg._vars, self._eg_f._vars)
        except:
            return self._eg._vars

    def addGeometry(self, dm: DifferentialMap, le: Lagrangian, g: Geometry):
        assert isinstance(dm, DifferentialMap)
        assert isinstance(le, Lagrangian)
        assert isinstance(g, Geometry)
        eg = WeightedGeometry(g=g, le=le).pull(dm)
        self._debugVars.append(eg.h())
        self._eg += eg
        self._refTrajs = joinRefTrajs(self._refTrajs, eg._refTrajs)

    def addWeightedGeometry(self, dm: DifferentialMap, eg: WeightedGeometry):
        assert isinstance(dm, DifferentialMap)
        assert isinstance(eg, WeightedGeometry)
        eg_pulled = eg.pull(dm)
        self._eg += eg_pulled
        self._refTrajs = joinRefTrajs(self._refTrajs, eg._refTrajs)

    def addForcingGeometry(
        self, dm: DifferentialMap, le: Lagrangian, g: Geometry, goalVelocity=None
    ):
        assert isinstance(dm, DifferentialMap)
        assert isinstance(le, Lagrangian)
        assert isinstance(g, Geometry)
        self._forcing = True
        if not hasattr(self, '_eg_f'):
            self._eg_f = deepcopy(self._eg)
        #eg_f = WeightedGeometry(g=g, le=le).pull(dm)
        self._eg_f += WeightedGeometry(g=g, le=le).pull(dm)
        self._refTrajs = self._eg_f._refTrajs
        # TODO: The following should be identitical <19-08-21, mspahn> #
        # self._eg_f += WeightedGeometry(g=g.pull(dm), le=le.pull(dm))
        self._vars = joinVariables(self._vars, self._eg_f._vars)
        if goalVelocity is not None:
            self._targetVelocity += ca.mtimes(ca.transpose(dm._J), goalVelocity)
        self._eg_f.concretize()

    def addForcingWeightedGeometry(
        self, dm: DifferentialMap, wg: WeightedGeometry, goalVelocity=None
    ):
        assert isinstance(dm, DifferentialMap)
        assert isinstance(wg, WeightedGeometry)
        self._forcing = True
        self._eg_f = deepcopy(self._eg)
        self._eg_f += wg.pull(dm)
        self._vars = joinVariables(self._vars, self._eg_f._vars)
        if goalVelocity is not None:
            self._targetVelocity += ca.mtimes(ca.pinv(dm._J), goalVelocity)
        self._eg_f.concretize()

    def setExecutionEnergy(self, lex: Lagrangian):
        assert isinstance(lex, Lagrangian)
        self._executionEnergy = True
        composed_geometry = Geometry(s=self._eg)
        self._eg_ex = WeightedGeometry(g=composed_geometry, le=lex)
        self._eg_ex.concretize()
        self._vars = joinVariables(self._vars, self._eg_ex._vars)
        if self._forcing:
            forced_geometry = Geometry(s=self._eg_f)
            self._eg_f_ex = WeightedGeometry(g=forced_geometry, le=lex)
            self._eg_f_ex.concretize()

    def concretize(self):
        self._eg.concretize()
        xddot = self._eg._xddot - self._eg._alpha * self._eg.xdot()
        if self._executionEnergy:
            xddot = self._eg_ex._xddot - self._eg_ex._alpha * self._eg_ex.xdot()
        if self._forcing:
            xddot = self._eg_f._xddot  # - self._eg_f._alpha * self._eg.xdot()
        if self._forcing and self._executionEnergy:
            xddot = self._eg_f_ex._xddot - self._eg_f_ex._alpha * self._eg.xdot()
        if self._speedControl:
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
            xddot = self._eg_f._xddot - (a_ex + beta_subst) * (
                self._eg.xdot()
                - ca.mtimes(self._eg_f.Minv(), self._targetVelocity)
            )
        totalVar = deepcopy(self._vars)
        for refTraj in self._refTrajs:
            totalVar += refTraj._vars
        self._funs = ca.Function("planner", totalVar, [xddot])
        if self._debug:
            # Put all variables you want to debug in here
            self._debugFuns = ca.Function("planner_debug", totalVar,
                self._debugVars
            )

    def computeAction(self, *args):
        for arg in args:
            assert isinstance(arg, np.ndarray)
        funs_val = self._funs(*args)
        action = np.array(funs_val)[:, 0]
        # avoid to small actions
        if np.linalg.norm(action) < eps:
            action = np.zeros(self._n)
        return action

    def debugEval(self, *args):
        for arg in args:
            assert isinstance(arg, np.ndarray)
        debugFuns_val = self._debugFuns(*args)
        res = []
        if not isinstance(debugFuns_val, tuple):
            return np.array(debugFuns_val)
        for val in debugFuns_val:
            if val.size()[1] == 1:
                res.append(np.array(val)[:, 0])
            else:
                res.append(np.array(val))
        return res

    def setSpeedControl(self, beta, eta):
        if not self._forcing:
            raise FabricPlannerException(
                "Speed control invalid",
                "Speed control cannot be set for unforced specs. Provide a"
                "valid forcing term through addForcingGeometry",
            )
        if not self._executionEnergy:
            raise FabricPlannerException(
                "Speed control invalid",
                "Speed control cannot be set without a"
                "given execution energy. Provide a valid execution"
                "energy through setExecutionEnergy",
            )
        self._speedControl = True
        self._constantSpeedControl = False
        self._beta = beta
        self._eta = eta

    def setDefaultSpeedControl(self, x_psi, dm_psi, exLag, ex_factor, **kwargs):
        p = {"a_b": 0.5, "r_b": 0.2, "b": [0.01, 6.5], "a_eta": 0.5, "a_shift": 0.5}
        for key in p.keys():
            if key in kwargs:
                p[key] = kwargs.get(key)
        s_beta = 0.5 * (ca.tanh(-p["a_b"] * (ca.norm_2(x_psi) - p["r_b"])) + 1)
        a_ex = ca.SX.sym("a_ex", 1)
        a_le = ca.SX.sym("a_le", 1)
        beta_fun = s_beta * p["b"][1] + p["b"][0] + ca.fmax(0, a_ex - a_le)
        beta = Damper(beta_fun, a_ex, a_le, x_psi, dm_psi)
        l_ex_d = ex_factor * exLag._l
        eta = 0.5 * (ca.tanh(-p["a_eta"] * (exLag._l - l_ex_d) - p["a_shift"]) + 1)
        self.setSpeedControl(beta, eta)

    def setConstantSpeedControl(self, beta=3.0):
        self._constant_beta = beta
        self._speedControl = True
        self._constantSpeedControl = True


class DefaultFabricPlanner(FabricPlanner):
    def __init__(self, n: int, **kwargs):
        q = ca.SX.sym("q", n)
        qdot = ca.SX.sym("qdot", n)
        p = {"m_base": 0.5, 'debug': False}
        for key in p.keys():
            if key in kwargs:
                p[key] = kwargs.get(key)
        # base geometry
        l_base = 0.5 * p["m_base"] * ca.dot(qdot, qdot)
        h_base = ca.SX(np.zeros(n))
        baseGeo = Geometry(h=h_base, x=q, xdot=qdot)
        baseLag = Lagrangian(l_base, x=q, xdot=qdot)
        super().__init__(baseGeo, baseLag, debug=p['debug'])