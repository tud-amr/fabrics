import pytest
import casadi as ca
import numpy as np
from fabrics.helpers.variables import Variables
from fabrics.diffGeometry.diffMap import DynamicDifferentialMap

Jdot_sign = +1


def test_dynamic_map_creation():
    x = ca.SX.sym("x", 2)
    xdot = ca.SX.sym("xdot", 2)
    x_ref = ca.SX.sym("x_ref", 2)
    xdot_ref = ca.SX.sym("xdot_ref", 2)
    xddot_ref = ca.SX.sym("xddot_ref", 2)

    state_variables = {'x': x, 'xdot': xdot}
    parameters = {'x_ref': x_ref, 'xdot_ref': xdot_ref, 'xddot_ref': xddot_ref}

    var = Variables(state_variables=state_variables, parameters=parameters)
    phi = x - x_ref
    phi_dot = xdot - xdot_ref
    Jdotqdot = -xddot_ref
    dm = DynamicDifferentialMap(var)
    dm.concretize()
    x_test = np.array([0.1, 0.2])
    xdot_test = np.array([0.1, 0.2])
    x_ref_test = np.array([0.5, 0.2])
    xdot_ref_test = np.array([-0.3, 1.2])
    xddot_ref_test = np.array([0.1, 0.2])
    x_rel, xdot_rel = dm.forward(
            x=x_test,
            xdot=xdot_test,
            x_ref=x_ref_test,
            xdot_ref=xdot_ref_test,
            xddot_ref=xddot_ref_test
        )
    assert isinstance(x_rel, np.ndarray)
    assert x_rel[0] == -0.4
    assert x_rel[1] == 0.0
    assert xdot_rel[0] == 0.4
