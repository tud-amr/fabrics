import pytest
import casadi as ca
import numpy as np
from optFabrics.diffGeometry.diffMap import DifferentialMap, RelativeDifferentialMap
from optFabrics.diffGeometry.variables import Jdot_sign
from optFabrics.diffGeometry.referenceTrajectory import ReferenceTrajectory


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
    q_rel = ca.SX.sym("q_rel", 2)
    qdot_rel = ca.SX.sym("qdot_rel", 2)
    refTraj = ReferenceTrajectory(2, ca.SX(np.identity(2)))
    phi = ca.norm_2(q - np.zeros(2))
    dm = DifferentialMap(phi, var=[q, qdot])
    dm_rel = RelativeDifferentialMap(var=[q, qdot], refTraj=refTraj)
    dm_var = DifferentialMap(ca.norm_2(q_rel), var=[q_rel, qdot_rel])
    return dm, dm_var, dm_rel


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
    dm, dm_var, dm_rel = variable_differentialMap
    dm.concretize()
    dm_var.concretize()
    dm_rel.concretize()
    q = np.array([1.0, -0.2])
    qdot = np.array([-0.3, 0.7])
    q_p = np.array([0.0, 0.0])
    qdot_p = np.array([0.0, 0.0])
    qddot_p = np.array([0.0, 0.0])
    q_rel, qdot_rel = dm_rel.forward(q, qdot, q_p, qdot_p, qddot_p)
    x, J, Jdot = dm.forward(q, qdot)
    x_var, J_var, Jdot_var = dm_var.forward(q_rel, qdot_rel)
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
    dm, dm_var, dm_rel = variable_differentialMap
    dm.concretize()
    dm_var.concretize()
    dm_rel.concretize()
    q = np.array([1.0, -0.2])
    qdot = np.array([-0.3, 0.7])
    q_p = np.array([-0.2, 0.2])
    qdot_p = np.array([-0.4, 0.8])
    qddot_p = np.array([-0.0, 0.0])
    q_rel, qdot_rel = dm_rel.forward(q, qdot, q_p, qdot_p, qddot_p)
    x_var, J_var, Jdot_var = dm_var.forward(q_rel, qdot_rel)
    xdot_var = np.dot(J_var, qdot_rel)
    J_test = 1 / np.linalg.norm(q - q_p) * np.array([q[0] - q_p[0], q[1] - q_p[1]])
    assert x_var[0] == pytest.approx(np.linalg.norm(q - q_p))
    assert J_var[0, 1] == pytest.approx(J_test[1])
    assert J_var[0, 0] == pytest.approx(J_test[0])
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
    assert xdot_var[0] == pytest.approx(xdot_p_test1 + xdot_p_test2)
