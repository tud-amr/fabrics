import casadi as ca

from fabrics.diffGeometry.diffMap import DifferentialMap
from fabrics.diffGeometry.energy import Lagrangian
from fabrics.helpers.functions import parse_symbolic_input


class Damper:
    def __init__(
        self, beta_expression: str, eta_expression: str, x: ca.SX, dm: DifferentialMap, lagrangian_execution: ca.SX
    ):
        assert isinstance(beta_expression, str)
        assert isinstance(eta_expression, str)
        assert isinstance(x, ca.SX)
        assert isinstance(dm, DifferentialMap)
        assert isinstance(lagrangian_execution, ca.SX)
        self._x = x
        self._dm = dm
        self._symbolic_parameters = {}
        self.parse_beta_expression(beta_expression)
        self.parse_eta_expression(eta_expression, lagrangian_execution)

    def parse_beta_expression(self, beta_expression):
        beta_parameters, self._beta = parse_symbolic_input(beta_expression, self._x, None, 'damper')
        if 'a_ex_damper' in beta_parameters:
            self._a_ex = beta_parameters['a_ex_damper']
            self._a_le = beta_parameters['a_le_damper']
            del(beta_parameters['a_ex_damper'])
            del(beta_parameters['a_le_damper'])
            self._constant_beta_expression = False
        else:
            self._constant_beta_expression = True
        self._symbolic_parameters.update(beta_parameters)

    def parse_eta_expression(self, eta_expression, lagrangian_execution):
        qdot = ca.vcat(ca.symvar(lagrangian_execution))
        eta_parameters, eta_raw = parse_symbolic_input(eta_expression, None, qdot, 'damper')
        if 'ex_lag_damper' in eta_parameters:
            ex_lag = eta_parameters['ex_lag_damper']
            self._eta = ca.substitute(eta_raw, ex_lag, lagrangian_execution)
            del(eta_parameters['ex_lag_damper'])
        else:
            self._eta = eta_raw
        self._symbolic_parameters.update(eta_parameters)

    def symbolic_parameters(self):
        return self._symbolic_parameters

    def substitute_beta(self, a_ex_fun, a_le_fun):
        if not self._constant_beta_expression:
            beta_subst = ca.substitute(self._beta, self._a_ex, a_ex_fun)
            beta_subst2 = ca.substitute(beta_subst, self._a_le, a_le_fun)
            beta_subst3 = ca.substitute(beta_subst2, self._x, self._dm._phi)
            return beta_subst3
        else:
            beta_subst = ca.substitute(self._beta, self._x, self._dm._phi)
            return beta_subst

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
