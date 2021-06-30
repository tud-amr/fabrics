import pytest
import casadi as ca
import numpy as np
from optFabrics.diffMap import DiffMap


def test_identy_map():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    x = ca.SX.sym("x", 2)
    xdot = ca.SX.sym("xdot", 2)
    phi = q
    dm = DiffMap("identity", phi, q, qdot, x, xdot)
    q_test = np.array([0.0, 1.0])
    qdot_test = np.array([1.0, 0.0])
    x, xdot, J, Jt, Jdot = dm.forwardMap(q_test, qdot_test, 0.0)
    for i in range(2):
        assert x[i] == q_test[i]

@pytest.fixture
def polar_map():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    x = ca.SX.sym("x", 2)
    xdot = ca.SX.sym("xdot", 2)
    phi = ca.vertcat(q[0] * ca.cos(q[1]), q[0] * ca.sin(q[1]))
    dm = DiffMap("identity", phi, q, qdot, x, xdot)
    return dm

def test_polar_map_1(polar_map):
    q_test = np.array([1.0, 0.0])
    qdot_test = np.array([1.0, 0.0])
    x, xdot, J, Jt, Jdot = polar_map.forwardMap(q_test, qdot_test, 0.0)
    assert x[0] == 1.0
    assert x[1] == 0.0
    assert xdot[0] == 1.0
    assert xdot[1] == 0.0

def test_polar_map_1(polar_map):
    q_test = np.array([2.0, 1.0])
    qdot_test = np.array([1.0, 0.0])
    x, xdot, J, Jt, Jdot = polar_map.forwardMap(q_test, qdot_test, 0.0)
    assert x[0] == pytest.approx(2*np.cos(1.0))
    assert x[1] == pytest.approx(2*np.sin(1.0))
    assert xdot[0] == pytest.approx(np.cos(1.0))
    assert xdot[1] == pytest.approx(np.sin(1.0))
