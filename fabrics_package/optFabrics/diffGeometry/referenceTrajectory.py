import casadi as ca
import numpy as np
from abc import ABC, abstractmethod

from optFabrics.diffGeometry.diffMap import DifferentialMap
from optFabrics.diffGeometry.variables import eps

class ReferenceTrajectory(ABC):
    def __init__(self, n: int, J: ca.SX):
        assert isinstance(n, int)
        assert isinstance(J, ca.SX)
        self._J = J
        self._n = n

    def n(self):
        return self._n

    def Jinv(self):
        import warnings
        warnings.warn("Casadi pseudo inverse is used in reference trajectory to transform velocity into joint space")
        Jt = ca.transpose(self._J)
        J = self._J
        JtJ = ca.mtimes(Jt, J)
        epsMatrix = ca.SX(np.identity(JtJ.size()[0]) * eps)
        return ca.mtimes(ca.pinv(JtJ + epsMatrix), Jt)

    @abstractmethod
    def concretize(self):
        pass

    @abstractmethod
    def evaluate(self, t):
        pass

    @abstractmethod
    def x(self):
        pass

    @abstractmethod
    def xdot(self):
        pass

    @abstractmethod
    def xddot(self):
        pass

    @abstractmethod
    def pull(self, dm: DifferentialMap):
        pass


class AnalyticTrajectory(ReferenceTrajectory):

    def __init__(self, n: int, J: ca.SX, **kwargs):
        super().__init__(n, J)
        if 'traj' in kwargs:
            self._traj = kwargs.get('traj')
            if isinstance(self._traj, list):
                self._traj = ca.vcat(self._traj)
            self._t = kwargs.get('t')
        self._n = n
        if 'var' in kwargs:
            self._vars = kwargs.get('var')
        else:
            suffix = 'ref'
            if 'name' in kwargs:
                suffix = kwargs.get('name')
            x = ca.SX.sym("x_" + suffix  ,n)
            xdot = ca.SX.sym("xdot_" + suffix  ,n)
            xddot = ca.SX.sym("xddot_" + suffix, n)
            self._vars = [x, xdot, xddot]

    def computeDerivatives(self):
        self._v = ca.jacobian(self._traj, self._t)
        self._a = ca.jacobian(self._v, self._t)

    def concretize(self):
        self.computeDerivatives()
        self._funs = ca.Function("traj", [self._t], [self._traj, self._v, self._a])

    def evaluate(self, t):
        fun = self._funs(t)
        x = np.array(fun[0])[:, 0]
        v = np.array(fun[1])[:, 0]
        a = np.array(fun[2])[:, 0]
        return x, v, a

    def x(self):
        return self._vars[0]

    def xdot(self):
        return ca.mtimes(self.Jinv(), self._vars[1])

    def xddot(self):
        return self._vars[2]

    def Jinv(self):
        import warnings
        warnings.warn("Casadi pseudo inverse is used in reference trajectory to transform velocity into joint space")
        Jt = ca.transpose(self._J)
        J = self._J
        JtJ = ca.mtimes(Jt, J)
        epsMatrix = ca.SX(np.identity(JtJ.size()[0]) * eps)
        return ca.mtimes(ca.pinv(JtJ + epsMatrix), Jt)

    def pull(self, dm: DifferentialMap):
        assert isinstance(dm, DifferentialMap)
        if hasattr(self, '_traj'):
            return AnalyticTrajectory(self._n, dm._J, traj=self._traj, t=self._t, var=self._vars)
        else:
            return AnalyticTrajectory(self._n, dm._J, var=self._vars)
