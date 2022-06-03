import pytest
import casadi as ca

from fabrics.helpers.functions import parse_symbolic_input

def test_parse_symbolic_input():
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    expression = "sym('alpha') * x**sym('exp')"
    new_parameters, symbolic_expression = parse_symbolic_input(expression, x, xdot, 'test')
    assert isinstance(new_parameters, dict)
    assert isinstance(symbolic_expression, ca.SX)
    assert 'alpha_test' in list(new_parameters.keys())
    assert 'exp_test' in list(new_parameters.keys())
