import casadi as ca
import numpy as np

from optFabrics.diffGeometry.diffMap import DifferentialMap
from optFabrics.diffGeometry.energy import Lagrangian
from optFabrics.diffGeometry.spec import Spec
from optFabrics.diffGeometry.geometry import Geometry
from optFabrics.diffGeometry.energized_geometry import WeightedGeometry, EnergizedGeometry


class FabricPlanner:
    """description"""

    def __init__(self, geo: Geometry, lag: Lagrangian):
        print("Initiializing fabric planner")
        assert isinstance(lag, Lagrangian)
        self._eg = WeightedGeometry(g=geo, le=lag)
        self._n = lag._x.size()[0]

    """
    def addTask(self, dm: DifferentialMap, le: Lagrangian, g: Geometry):
        assert isinstance(dm, DifferentialMap)
        assert isinstance(le, Lagrangian)
        assert isinstance(g, Geometry)
        s_energized = EnergizedGeometry(g, le)
        s_pulled = s_energized.pull(dm)
        self._spec += s_pulled
    """

    def addGeometry(self, dm: DifferentialMap, le: Lagrangian, g: Geometry):
        assert isinstance(dm, DifferentialMap)
        assert isinstance(le, Lagrangian)
        assert isinstance(g, Geometry)
        self._eg += WeightedGeometry(g=g, le=le).pull(dm)

    def setExecutionEnergy(self, lex: Lagrangian):
        composed_geometry = Geometry(s=self._eg)
        self._eg_ex = WeightedGeometry(g=composed_geometry, le=lex)
        self._eg_ex.concretize()

    def concretize(self):
        self._eg.concretize()

    def computeAction(self, x, xdot):
        M, f, xddot, alpha = self._eg.evaluate(x, xdot)
        return xddot - alpha * xdot

    def computeActionEx(self, x, xdot):
        _, _, xddot, alpha_ex = self._eg_ex.evaluate(x, xdot)
        return xddot - alpha_ex * xdot
