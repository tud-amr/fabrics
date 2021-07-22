import casadi as ca
import numpy as np

from optFabrics.diffGeometry.diffMap import DifferentialMap
from optFabrics.diffGeometry.energy import FinslerStructure, Lagrangian
from optFabrics.diffGeometry.spec import Spec
from optFabrics.diffGeometry.geometry import Geometry
from optFabrics.diffGeometry.energized_geometry import EnergizedGeometry

class FabricPlanner:
    """description"""

    def __init__(self, x : ca.SX, xdot : ca.SX, m=1.0):
        print("Initiializing fabric planner")
        self._n = x.size()[0]
        M = ca.SX(np.identity(self._n) * m)
        f = ca.SX(np.zeros((self._n, 1)))
        self._spec = Spec(M, f, x, xdot)

    def addTask(self, dm : DifferentialMap, le : Lagrangian, g : Geometry):
        assert isinstance(dm, DifferentialMap)
        assert isinstance(le, Lagrangian)
        assert isinstance(g, Geometry)
        s_energized = EnergizedGeometry(g, le)
        s_pulled = s_energized.pull(dm)
        self._spec += s_pulled

    def concretize(self):
        self._spec.concretize()

    def computeAction(self, x, xdot):
        M, f, xddot = self._spec.evaluate(x, xdot)
        return xddot
