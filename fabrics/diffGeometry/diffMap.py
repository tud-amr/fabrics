import casadi as ca
import numpy as np

from fabrics.helpers.casadiFunctionWrapper import CasadiFunctionWrapper
from fabrics.helpers.variables import Variables


class DifferentialMap:
    """description"""

    def __init__(self, phi: ca.SX, **kwargs):
        if 'q' in kwargs.keys() and 'qdot' in kwargs.keys():
            q = kwargs.get('q')
            qdot = kwargs.get('qdot')
            self._vars = Variables(state_variables={'q': q, 'qdot': qdot})
        elif 'var' in kwargs.keys():
            self._vars = kwargs.get('var')
        Jdot_sign = -1
        if 'Jdot_sign' in kwargs.keys():
            Jdot_sign = kwargs.get('Jdot_sign')
        assert isinstance(phi, ca.SX)
        self._vars.verify()
        self._phi = phi
        q = self._vars.variable_by_name('q')
        qdot = self._vars.variable_by_name('qdot')
        self._J = ca.jacobian(phi, q)
        self._Jdot = Jdot_sign * ca.jacobian(ca.mtimes(self._J, qdot), q)

    def Jdotqdot(self):
        return ca.mtimes(self._Jdot, self.qdot())

    def phidot(self):
        return ca.mtimes(self._J, self.qdot())

    def concretize(self):
        self._funs = CasadiFunctionWrapper(
            "funs", self._vars.asDict(), {"phi": self._phi, "J": self._J, "Jdot": self._Jdot}
        )

    def params(self):
        return []

    def forward(self, values):
        for key in values:
            assert isinstance(values[key], np.ndarray)
        funs = self._funs.evaluate(values)
        x = np.array(funs['phi'])[:, 0]
        J = np.array(funs['J'])
        Jdot = np.array(funs['Jdot'])
        return x, J, Jdot

    def q(self):
        return self._vars.variable_by_name('q')

    def qdot(self):
        return self._vars.variable_by_name('qdot')

class ParameterizedDifferentialMap(DifferentialMap):
    def __init__(self, phi: ca.SX, params, **kwargs):
        super().__init__(phi, **kwargs)
        self._params = params

    def params(self):
        return self._params


class RelativeDifferentialMap(DifferentialMap):
    def __init__(self, **kwargs):
        if 'q' in kwargs.keys() and 'qdot' in kwargs.keys():
            q = kwargs.get('q')
            qdot = kwargs.get('qdot')
            self._vars = Variables(state_variables={'q': q, 'qdot': qdot})
        elif 'var' in kwargs.keys():
            self._vars = kwargs.get('var')
        if 'refTraj' in kwargs:
            self._refTraj = kwargs.get('refTraj')
        phi = self._vars.variable_by_name('q') - self._refTraj.x()
        super().__init__(phi, var=self._vars)

    def forward(self, values):
        for key in values:
            assert isinstance(values[key], np.ndarray)
        funs = self._funs.evaluate(values)
        x = np.array(funs['phi'])[:, 0]
        xdot = values['qdot'] - values['xdot']
        return x, xdot

    def concretize(self):
        var = self._vars + self._refTraj._vars
        self._funs = CasadiFunctionWrapper(
            "funs", var.asDict(), {"phi": self._phi, "J": self._J, "Jdot": self._Jdot}
        )

    def Jdotqdot(self):
        return -1 * self._refTraj.xddot()

    def phidot(self):
        return self.qdot() - self._refTraj.xdot()
