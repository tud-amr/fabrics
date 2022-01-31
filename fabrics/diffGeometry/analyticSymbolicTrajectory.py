import casadi as ca

from fabrics.diffGeometry.diffMap import DifferentialMap
from fabrics.diffGeometry.symbolicTrajectory import SymbolicTrajectory
from MotionPlanningSceneHelpers.analyticTrajectory import AnalyticTrajectory


class AnalyticSymbolicTrajectory(SymbolicTrajectory):

    def __init__(self, J: ca.SX, n: int, **kwargs):
        if 'traj' not in kwargs:
            refTraj = AnalyticTrajectory(n, traj=['0', ] * n)
        elif isinstance(kwargs.get('traj'), list):
            refTraj = AnalyticTrajectory(n, **kwargs)
        else:
            refTraj = kwargs.get('traj')
        super().__init__(refTraj, J, **kwargs)

    def pull(self, dm: DifferentialMap):
        assert isinstance(dm, DifferentialMap)
        if hasattr(self, '_refTraj'):
            return AnalyticSymbolicTrajectory(dm._J, self.n(), traj=self._refTraj, var=self._vars)
        else:
            return AnalyticSymbolicTrajectory(dm._J, self.n(), var=self._vars)
