import casadi as ca
import numpy as np


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
        self._Jdot = ca.jacobian(ca.mtimes(self._J, qdot), q)

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
