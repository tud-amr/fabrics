import casadi as ca
import numpy as np


class DifferentialMap:
    """description"""

    def __init__(self, phi: ca.SX, **kwargs):
        if 'q' in kwargs.keys() and 'qdot' in kwargs.keys():
            q = kwargs.get('q')
            qdot = kwargs.get('qdot')
        elif 'var' in kwargs.keys():
            q, qdot = kwargs.get('var')
        Jdot_sign = -1
        if 'Jdot_sign' in kwargs.keys():
            Jdot_sign = kwargs.get('Jdot_sign')
        assert isinstance(q, ca.SX)
        assert isinstance(qdot, ca.SX)
        assert isinstance(phi, ca.SX)
        self._vars = [q, qdot]
        self._phi = phi
        self._J = ca.jacobian(phi, q)
        self._Jdot = Jdot_sign * ca.jacobian(ca.mtimes(self._J, qdot), q)

    def Jdotqdot(self):
        return ca.mtimes(self._Jdot, self.qdot())

    def phidot(self):
        return ca.mtimes(self._J, self.qdot())

    def concretize(self):
        self._fun = ca.Function(
            "forward", self._vars, [self._phi, self._J, self._Jdot]
        )

    def forward(self, q: np.ndarray, qdot: np.ndarray):
        assert isinstance(q, np.ndarray)
        assert isinstance(qdot, np.ndarray)
        funs = self._fun(q, qdot)
        x = np.array(funs[0])[:, 0]
        J = np.array(funs[1])
        Jdot = np.array(funs[2])
        return x, J, Jdot

    def q(self):
        return self._vars[0]

    def qdot(self):
        return self._vars[1]


class RelativeDifferentialMap(DifferentialMap):
    def __init__(self, **kwargs):
        if 'q' in kwargs:
            q = kwargs.get('q')
            qdot = kwargs.get('qdot')
        elif 'var' in kwargs:
            q, qdot = kwargs.get('var')
        if 'refTraj' in kwargs:
            self._refTraj = kwargs.get('refTraj')
        phi = q - self._refTraj.x()
        super().__init__(phi, q=q, qdot=qdot)

    def forward(self, q: np.ndarray, qdot: np.ndarray, q_p: np.ndarray, qdot_p: np.ndarray, qddot_p: np.ndarray):
        assert isinstance(q, np.ndarray)
        assert isinstance(qdot, np.ndarray)
        assert isinstance(q_p, np.ndarray)
        assert isinstance(qdot_p, np.ndarray)
        assert isinstance(qddot_p, np.ndarray)
        funs = self._fun(q, qdot, q_p, qdot_p, qddot_p)
        x = np.array(funs[0])[:, 0]
        xdot = qdot - qdot_p
        return x, xdot

    def concretize(self):
        var = self._vars + self._refTraj._vars
        self._fun = ca.Function(
            "forward", var, [self._phi, self._J, self._Jdot]
        )

    def Jdotqdot(self):
        return -1 * self._refTraj.xddot()

    def phidot(self):
        return self.qdot() - self._refTraj.xdot()
