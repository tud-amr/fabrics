import casadi as ca
import numpy as np
from copy import deepcopy

from optFabrics.diffGeometry.diffMap import DifferentialMap
from optFabrics.diffGeometry.energy import Lagrangian
from optFabrics.diffGeometry.spec import Spec
from optFabrics.diffGeometry.geometry import Geometry
from optFabrics.diffGeometry.energized_geometry import (
    WeightedGeometry,
    EnergizedGeometry,
)


class FabricPlannerException(Exception):
    def __init__(self, expression, message):
        self._expression = expression
        self._message = message

    def what(self):
        return self._expression + ": " + self._message


class FabricPlanner:
    """description"""

    def __init__(self, geo: Geometry, lag: Lagrangian):
        print("Initiializing fabric planner")
        assert isinstance(lag, Lagrangian)
        self._eg = WeightedGeometry(g=geo, le=lag)
        self._n = lag._x.size()[0]
        self._forcing = False
        self._executionEnergy = False
        self._speedControl = False

    def addGeometry(self, dm: DifferentialMap, le: Lagrangian, g: Geometry):
        assert isinstance(dm, DifferentialMap)
        assert isinstance(le, Lagrangian)
        assert isinstance(g, Geometry)
        self._eg += WeightedGeometry(g=g, le=le).pull(dm)

    def addForcingGeometry(self, dm: DifferentialMap, le: Lagrangian, g: Geometry):
        assert isinstance(dm, DifferentialMap)
        assert isinstance(le, Lagrangian)
        assert isinstance(g, Geometry)
        self._forcing = True
        self._eg_f = deepcopy(self._eg)
        self._eg_f += WeightedGeometry(g=g, le=le).pull(dm)
        self._eg_f.concretize()

    def setExecutionEnergy(self, lex: Lagrangian):
        assert isinstance(lex, Lagrangian)
        self._executionEnergy = True
        composed_geometry = Geometry(s=self._eg)
        self._eg_ex = WeightedGeometry(g=composed_geometry, le=lex)
        self._eg_ex.concretize()
        if self._forcing:
            forced_geometry = Geometry(s=self._eg_f)
            self._eg_f_ex = WeightedGeometry(g=forced_geometry, le=lex)
            self._eg_f_ex.concretize()

    def concretize(self):
        self._eg.concretize()
        xddot = self._eg._xddot - self._eg._alpha * self._eg._xdot
        if self._executionEnergy:
            xddot = self._eg_ex._xddot - self._eg_ex._alpha * self._eg._xdot
        if self._forcing:
            xddot = self._eg_f._xddot - self._eg_f._alpha * self._eg._xdot
        if self._forcing and self._executionEnergy:
            xddot = self._eg_f_ex._xddot - self._eg_f_ex._alpha * self._eg._xdot
        if self._speedControl:
            a_ex = self._eta * self._eg._alpha + (1 - self._eta) * self._eg_f_ex._alpha
            beta_subst = self._beta.substitute(-a_ex, -self._eg._alpha)
            xddot = (
                self._eg_f_ex._xddot
                - a_ex * self._eg._xdot
                - beta_subst * self._eg._xdot
            )
        self._funs = ca.Function("planner", [self._eg._x, self._eg._xdot], [xddot])

    def computeAction(self, x, xdot):
        return np.array(self._funs(x, xdot))[:, 0]

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
        self._beta = beta
        self._eta = eta
