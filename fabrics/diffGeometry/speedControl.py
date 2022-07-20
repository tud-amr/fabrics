import casadi as ca

from fabrics.diffGeometry.diffMap import DifferentialMap
from fabrics.diffGeometry.energy import Lagrangian
from fabrics.helpers.functions import parse_symbolic_input


class Damper:
    def __init__(
        self, beta_expression: str, eta_expression: str, a_ex: ca.SX, a_le: ca.SX, x: ca.SX, dm: DifferentialMap, lagrangian_execution: ca.SX
    ):
        assert isinstance(beta_expression, str)
        assert isinstance(eta_expression, str)
        assert isinstance(a_ex, ca.SX)
        assert isinstance(a_le, ca.SX)
        assert isinstance(x, ca.SX)
        assert isinstance(dm, DifferentialMap)
        assert isinstance(lagrangian_execution, ca.SX)
        new_parameters, self._beta = parse_symbolic_input(beta_expression, x, None, 'damper')
        new_parameters_eta, eta_raw = parse_symbolic_input(eta_expression, None, None, 'damper')
        if 'ex_lag_damper' in new_parameters_eta:
            ex_lag = new_parameters_eta['ex_lag_damper']
            self._eta = ca.substitute(eta_raw, ex_lag, lagrangian_execution)
            del(new_parameters_eta['ex_lag_damper'])
        else:
            self._eta = eta_raw
        self._a_ex = new_parameters['a_ex_damper']
        self._a_le = new_parameters['a_le_damper']
        del(new_parameters['a_ex_damper'])
        del(new_parameters['a_le_damper'])
        self._x = x
        self._dm = dm
        self._symbolic_parameters = {}
        self._symbolic_parameters.update(new_parameters)
        self._symbolic_parameters.update(new_parameters_eta)

    def symbolic_parameters(self):
        return self._symbolic_parameters

    def substitute_beta(self, a_ex_fun, a_le_fun):
        beta_subst = ca.substitute(self._beta, self._a_ex, a_ex_fun)
        beta_subst2 = ca.substitute(beta_subst, self._a_le, a_le_fun)
        beta_subst3 = ca.substitute(beta_subst2, self._x, self._dm._phi)
        return beta_subst3

    def substitute_eta(self):
        return self._eta

class Interpolator:
    def __init__(self, eta: ca.SX, lex: Lagrangian, lex_d: Lagrangian):
        assert isinstance(eta, ca.SX)
        assert isinstance(lex, Lagrangian)
        assert isinstance(lex_d, Lagrangian)
        self._eta = eta
        self._lex = lex
        self._lex_d = lex_d
