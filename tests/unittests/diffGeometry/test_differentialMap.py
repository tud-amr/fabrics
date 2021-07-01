import pytest
import casadi as ca
import numpy as np
from optFabrics.diffGeometry.diffMap import DifferentialMap

def test_dm_creation():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    phi = ca.vertcat(q[0] * ca.cos(q[1]), q[0] * ca.sin(q[1]))
    dm = DifferentialMap(q, qdot, phi)
    assert ca.is_equal(q, dm._q)

@pytest.fixture
def simple_differentialMap():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    phi = ca.vertcat(q[0] * ca.cos(q[1]), q[0] * ca.sin(q[1]))
    dm = DifferentialMap(q, qdot, phi)
    return dm

def test_forward_mapping_polar(simple_differentialMap):
    q = np.array([1.0, -0.3])
    qdot = np.array([0.3, 1.1])
    simple_differentialMap.concretize()
    x, J, Jdot = simple_differentialMap.forward(q, qdot)
    assert isinstance(x, np.ndarray)
    assert isinstance(J, np.ndarray)
    assert isinstance(Jdot, np.ndarray)
    assert x.size == 2
    assert J.shape == (2, 2)
    assert Jdot.shape == (2, 2)
    assert x[0] == q[0] * np.cos(q[1])
    assert x[1] == q[0] * np.sin(q[1])
    assert J[0, 0] == np.cos(q[1])
    assert J[0, 1] == -np.sin(q[1]) * q[0]
    assert J[1, 0] == np.sin(q[1])
    assert J[1, 1] == np.cos(q[1]) * q[0]
    assert Jdot[0, 0] == -np.sin(q[1]) * qdot[1]
    assert Jdot[0, 1] == -np.sin(q[1]) * qdot[0] - q[0] * np.cos(q[1]) * qdot[1]
    assert Jdot[1, 0] == np.cos(q[1]) * qdot[1]
    assert Jdot[1, 1] == np.cos(q[1]) * qdot[0] - q[0] * np.sin(q[1]) * qdot[1]
