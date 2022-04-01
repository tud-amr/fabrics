import casadi as ca
import numpy as np
from abc import ABC, abstractmethod

from fabrics.diffGeometry.diffMap import DifferentialMap
from fabrics.helpers.constants import eps
from fabrics.helpers.variables import Variables
from MotionPlanningSceneHelpers.referenceTrajectory import ReferenceTrajectory


class SymbolicTrajectory(ABC):
    def __init__(self, refTraj: ReferenceTrajectory, J: ca.SX, **kwargs):
        assert isinstance(J, ca.SX)
        assert isinstance(refTraj, ReferenceTrajectory)
        self._J = J
        self._refTraj = refTraj
        if 'var' in kwargs:
            self._vars = kwargs.get('var')
        else:
            suffix = kwargs.get('name') if 'name' in kwargs else 'ref'
            x = ca.SX.sym("x_" + suffix, self.n())
            xdot = ca.SX.sym("xdot_" + suffix, self.n())
            xddot = ca.SX.sym("xddot_" + suffix, self.n())
            self._vars = Variables(parameters={'x': x, 'xdot': xdot, 'xddot': xddot})

    def n(self):
        return self._refTraj.n()

    def Jinv(self):
        import warnings
        warnings.warn("Casadi pseudo inverse is used in reference trajectory to transform velocity into joint space")
        Jt = ca.transpose(self._J)
        J = self._J
        JtJ = ca.mtimes(Jt, J)
        epsMatrix = ca.SX(np.identity(JtJ.size()[0]) * eps)
        return ca.mtimes(ca.pinv(JtJ + epsMatrix), Jt)

    def concretize(self):
        self._refTraj.concretize()

    def evaluate(self, t):
        return self._refTraj.evaluate(t)

    def x(self):
        return self._vars.parameter_by_name('x')

    def xdot(self):
        return ca.mtimes(self.Jinv(), self._vars.parameter_by_name('xdot'))

    def xddot(self):
        return self._vars.parameter_by_name('xddot')

    @abstractmethod
    def pull(self, dm: DifferentialMap):
        pass


