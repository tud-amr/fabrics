import casadi as ca
import numpy as np

from optFabrics.diffGeometry.variables import Jdot_sign


class DifferentialMap:
    """description"""

    def __init__(self, q: ca.SX, qdot: ca.SX, phi: ca.SX):
        assert isinstance(q, ca.SX)
        assert isinstance(qdot, ca.SX)
        assert isinstance(phi, ca.SX)
        self._q = q
        self._qdot = qdot
        self._phi = phi
        self._J = ca.jacobian(phi, q)
        self._Jdot = Jdot_sign * ca.jacobian(ca.mtimes(self._J, qdot), q)

    def concretize(self):
        self._fun = ca.Function(
            "forward", [self._q, self._qdot], [self._phi, self._J, self._Jdot]
        )

    def forward(self, q: np.ndarray, qdot: np.ndarray):
        assert isinstance(q, np.ndarray)
        assert isinstance(qdot, np.ndarray)
        funs = self._fun(q, qdot)
        x = np.array(funs[0])[:, 0]
        J = np.array(funs[1])
        Jdot = np.array(funs[2])
        return x, J, Jdot


class VariableDifferentialMap(DifferentialMap):
    def __init__(self, q: ca.SX, qdot: ca.SX, phi: ca.SX, q_p: ca.SX, qdot_p: ca.SX):
        super().__init__(q, qdot, phi)
        self._q_p = q_p
        self._qdot_p = qdot_p
        self._J_p = ca.jacobian(phi, q_p)
        self._Jdot_p = Jdot_sign * ca.jacobian(ca.mtimes(self._J_p, qdot_p), q_p)

    def concretize(self):
        self._fun = ca.Function(
            "forward", [self._q, self._qdot, self._q_p, self._qdot_p], [self._phi, self._J, self._Jdot, self._J_p, self._Jdot_p]
        )

    def forward(self, q: np.ndarray, qdot: np.ndarray, q_p: np.ndarray, qdot_p: np.ndarray):
        assert isinstance(q, np.ndarray)
        assert isinstance(qdot, np.ndarray)
        assert isinstance(q_p, np.ndarray)
        assert isinstance(qdot_p, np.ndarray)
        funs = self._fun(q, qdot, q_p, qdot_p)
        x = np.array(funs[0])[:, 0]
        J = np.array(funs[1])
        Jdot = np.array(funs[2])
        J_p = np.array(funs[3])
        Jdot_p = np.array(funs[4])
        return x, J, Jdot, J_p, Jdot_p
