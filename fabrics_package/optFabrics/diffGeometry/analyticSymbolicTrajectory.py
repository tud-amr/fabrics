import casadi as ca

from optFabrics.diffGeometry.diffMap import DifferentialMap
from optFabrics.diffGeometry.symbolicTrajectory import SymbolicTrajectory
from MotionPlanningSceneHelpers.analyticTrajectory import AnalyticTrajectory


class AnalyticSymbolicTrajectory(SymbolicTrajectory):

    def __init__(self, J: ca.SX, n: int, **kwargs):
        if 'traj' not in kwargs:
            refTraj = AnalyticTrajectory(n, traj=['0', ] * n)
        elif not isinstance(kwargs.get('traj'), AnalyticTrajectory):
            refTraj = AnalyticTrajectory(n, **kwargs)
        else:
            refTraj = kwargs.get('traj')
        super().__init__(refTraj, J, **kwargs)

    def pull(self, dm: DifferentialMap):
        assert isinstance(dm, DifferentialMap)
        if hasattr(self, '_refTraj'):
            return AnalyticSymbolicTrajectory(dm._J, self.n(), traj=self._refTraj, var=self._vars)
        else:
            __import__('pdb').set_trace()
            return AnalyticSymbolicTrajectory(dm._J, self.n(), var=self._vars)
