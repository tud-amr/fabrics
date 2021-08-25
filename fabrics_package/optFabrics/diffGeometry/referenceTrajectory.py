import casadi as ca
import numpy as np

from optFabrics.diffGeometry.diffMap import DifferentialMap


class ReferenceTrajectory:

    def __init__(self, n: int, J: ca.SX, **kwargs):
        assert isinstance(n, int)
        assert isinstance(J, ca.SX)
        if 'traj' in kwargs:
            self._traj = kwargs.get('traj')
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
        self._J = J

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
        return ca.mtimes(ca.pinv(self._J), self._vars[1])

    def xddot(self):
        return self._vars[2]

    def pull(self, dm: DifferentialMap):
        assert isinstance(dm, DifferentialMap)
        if hasattr(self, '_traj'):
            return ReferenceTrajectory(self._n, dm._J, traj=self._traj, t=self._t, var=self._vars)
        else:
            return ReferenceTrajectory(self._n, dm._J, var=self._vars)
