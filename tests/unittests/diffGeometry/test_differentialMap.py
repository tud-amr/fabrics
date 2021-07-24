import pytest
import casadi as ca
import numpy as np
from optFabrics.diffGeometry.diffMap import DifferentialMap, VariableDifferentialMap
from optFabrics.diffGeometry.variables import Jdot_sign


def test_dm_creation():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    phi = ca.vertcat(q[0] * ca.cos(q[1]), q[0] * ca.sin(q[1]))
    dm = DifferentialMap(phi, q=q, qdot=qdot)
    assert ca.is_equal(q, dm.q())


@pytest.fixture
def simple_differentialMap():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    phi = ca.vertcat(q[0] * ca.cos(q[1]), q[0] * ca.sin(q[1]))
    dm = DifferentialMap(phi, q=q, qdot=qdot)
    return dm


@pytest.fixture
def variable_differentialMap():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    q_p = ca.SX.sym("q_p", 2)
    qdot_p = ca.SX.sym("qdot_p", 2)
    phi = ca.norm_2(q - np.zeros(2))
    phi_var = ca.norm_2(q - q_p)
    dm = DifferentialMap(phi, var=[q, qdot])
    dm_var = VariableDifferentialMap(phi_var, var=[q, qdot, q_p, qdot_p])
    return dm, dm_var


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
    Jdot_test = Jdot_sign * np.array(
        [
            [
                -np.sin(q[1]) * qdot[1],
                -np.sin(q[1]) * qdot[0] - q[0] * np.cos(q[1]) * qdot[1],
            ],
            [
                np.cos(q[1]) * qdot[1],
                np.cos(q[1]) * qdot[0] - q[0] * np.sin(q[1]) * qdot[1],
            ],
        ]
    )
    assert Jdot[0, 0] == pytest.approx(Jdot_test[0, 0])
    assert Jdot[0, 1] == pytest.approx(Jdot_test[0, 1])
    assert Jdot[1, 0] == pytest.approx(Jdot_test[1, 0])
    assert Jdot[1, 1] == pytest.approx(Jdot_test[1, 1])


def test_variable_map_Zero(variable_differentialMap):
    dm = variable_differentialMap[0]
    dm_var = variable_differentialMap[1]
    dm.concretize()
    dm_var.concretize()
    q = np.array([1.0, -0.2])
    qdot = np.array([-0.3, 0.7])
    q_p = np.array([0.0, 0.0])
    qdot_p = np.array([0.0, 0.0])
    x, J, Jdot = dm.forward(q, qdot)
    x_var, J_var, Jdot_var, J_p, Jdot_p = dm_var.forward(q, qdot, q_p, qdot_p)
    xdot = np.dot(J, qdot)
    xdot_var = np.dot(J_var, qdot)
    assert x[0] == pytest.approx(np.linalg.norm(q - q_p))
    assert x_var[0] == pytest.approx(np.linalg.norm(q - q_p))
    assert J[0, 0] == pytest.approx(J_var[0, 0])
    assert Jdot[0, 1] == pytest.approx(Jdot_var[0, 1])
    assert Jdot[0, 0] == pytest.approx(Jdot_var[0, 0])
    assert J[0, 1] == pytest.approx(J_var[0, 1])
    xdot_test = 1 / np.linalg.norm(q) * (q[0] * qdot[0] + q[1] * qdot[1])
    assert xdot[0] == pytest.approx(xdot_test)
    assert xdot_var[0] == pytest.approx(xdot_test)


def test_variable_map_NonZero(variable_differentialMap):
    dm_var = variable_differentialMap[1]
    dm_var.concretize()
    q = np.array([1.0, -0.2])
    qdot = np.array([-0.3, 0.7])
    q_p = np.array([-0.2, 0.2])
    qdot_p = np.array([-0.4, 0.8])
    x, J, Jdot, J_p, Jdot_p = dm_var.forward(q, qdot, q_p, qdot_p)
    xdot = np.dot(J, qdot) + np.dot(J_p, qdot_p)
    J_test = 1 / np.linalg.norm(q - q_p) * np.array([q[0] - q_p[0], q[1] - q_p[1]])
    Jdot_test = Jdot_sign * (
        1 / np.linalg.norm(q - q_p) * qdot
        - 1 / (np.linalg.norm(q - q_p) ** 3) * np.dot(q - q_p, qdot) * (q - q_p)
    )
    Jdot_p_test = Jdot_sign * (
        + 1 / np.linalg.norm(q - q_p) * qdot_p
        - 1 / (np.linalg.norm(q - q_p) ** 3) * np.dot(q - q_p, qdot_p) * (q - q_p)
    )
    assert x[0] == pytest.approx(np.linalg.norm(q - q_p))
    assert J[0, 1] == pytest.approx(J_test[1])
    assert J[0, 0] == pytest.approx(J_test[0])
    assert Jdot[0, 1] == pytest.approx(Jdot_test[1])
    assert Jdot[0, 0] == pytest.approx(Jdot_test[0])
    assert Jdot_p[0, 1] == pytest.approx(Jdot_p_test[1])
    assert Jdot_p[0, 0] == pytest.approx(Jdot_p_test[0])
    xdot_p_test1 = (
        1
        / np.linalg.norm(q - q_p)
        * ((q[0] - q_p[0]) * qdot[0] + (q[1] - q_p[1]) * qdot[1])
    )
    xdot_p_test2 = (
        -1
        / np.linalg.norm(q - q_p)
        * ((q[0] - q_p[0]) * qdot_p[0] + (q[1] - q_p[1]) * qdot_p[1])
    )
    assert xdot[0] == pytest.approx(xdot_p_test1 + xdot_p_test2)
