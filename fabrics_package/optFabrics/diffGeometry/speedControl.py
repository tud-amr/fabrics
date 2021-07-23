import casadi as ca

from optFabrics.diffGeometry.diffMap import DifferentialMap
from optFabrics.diffGeometry.energy import Lagrangian


class Damper:
    def __init__(
        self, beta: ca.SX, a_ex: ca.SX, a_le: ca.SX, x: ca.SX, dm: DifferentialMap
    ):
        assert isinstance(beta, ca.SX)
        assert isinstance(a_ex, ca.SX)
        assert isinstance(a_le, ca.SX)
        assert isinstance(x, ca.SX)
        assert isinstance(dm, DifferentialMap)
        self._beta = beta
        self._a_ex = a_ex
        self._a_le = a_le
        self._x = x
        self._dm = dm

    def substitute(self, a_ex_fun, a_le_fun):
        beta_subst = ca.substitute(self._beta, self._a_ex, a_ex_fun)
        beta_subst2 = ca.substitute(beta_subst, self._a_le, a_le_fun)
        beta_subst3 = ca.substitute(beta_subst2, self._x, self._dm._phi)
        return beta_subst3


class Interpolator:
    def __init__(self, eta: ca.SX, lex: Lagrangian, lex_d: Lagrangian):
        assert isinstance(eta, ca.SX)
        assert isinstance(lex, Lagrangian)
        assert isinstance(lex_d, Lagrangian)
        self._eta = eta
        self._lex = lex
        self._lex_d = lex_d
