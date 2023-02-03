import pytest
import casadi as ca
import numpy as np
from fabrics.diffGeometry.diffMap import DifferentialMap, DynamicDifferentialMap
from fabrics.helpers.variables import Variables

Jdot_sign = +1


def test_dm_creation():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    phi = ca.vertcat(q[0] * ca.cos(q[1]), q[0] * ca.sin(q[1]))
    variables = Variables(state_variables={'q': q, 'qdot': qdot})
    dm = DifferentialMap(phi, variables)
    assert ca.is_equal(q, dm.q())


@pytest.fixture
def simple_differentialMap():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    phi = ca.vertcat(q[0] * ca.cos(q[1]), q[0] * ca.sin(q[1]))
    variables = Variables(state_variables={'q': q, 'qdot': qdot})
    dm = DifferentialMap(phi, variables=variables, Jdot_sign=Jdot_sign)
    return dm


@pytest.fixture
def variable_differentialMap():
    """
            -----------------
            |     q,qdot    |
            -----------------
                    |
                    v
            -----------------
            |q_rel,qdot_rel |
            -----------------
                    |
                    v
            -----------------
            |     x,xdot    |
            -----------------
    """
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    q_rel = ca.SX.sym("q_rel", 2)
    qdot_rel = ca.SX.sym("qdot_rel", 2)
    q_ref = ca.SX.sym("q_ref", 2)
    qdot_ref = ca.SX.sym("qdot_ref", 2)
    qddot_ref = ca.SX.sym("qddot_ref", 2)
    var_dynamic = Variables(state_variables={'q': q, 'qdot': qdot}, parameters={'x_ref': q_ref, 'xdot_ref': qdot_ref, 'xddot_ref': qddot_ref})
    dm_dynamic = DynamicDifferentialMap(var_dynamic)
    phi_static = ca.norm_2(q_rel)
    var_static = Variables(state_variables={'q_rel': q_rel, 'qdot_rel': qdot_rel})
    dm_static = DifferentialMap(phi_static, var_static)
    return dm_static, dm_dynamic



def test_forward_mapping_polar(simple_differentialMap):
    q = np.array([1.0, -0.3])
    qdot = np.array([0.3, 1.1])
    simple_differentialMap.concretize()
    x, J, Jdot = simple_differentialMap.forward(q=q, qdot=qdot)
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
    dm_static, dm_dynamic = variable_differentialMap
    dm_static.concretize()
    dm_dynamic.concretize()
    q = np.array([1.0, -0.2])
    qdot = np.array([-0.3, 0.7])
    q_p = np.array([0.0, 0.0])
    qdot_p = np.array([0.0, 0.0])
    qddot_p = np.array([0.0, 0.0])
    q_rel, qdot_rel = dm_dynamic.forward(q= q, qdot=qdot, x_ref=q_p, xdot_ref=qdot_p, xddot_ref=qddot_p)
    x, J, Jdot = dm_static.forward(q_rel=q_rel, qdot_rel=qdot_rel)
    J_test = 1/np.linalg.norm(q-q_p) * (q - q_p)
    Jdot_test = [-0.120687, -0.603434]
    assert x[0] == pytest.approx(np.linalg.norm(q - q_p))
    assert J[0, 0] == pytest.approx(J_test[0])
    assert J[0, 1] == pytest.approx(J_test[1])
    assert Jdot[0, 0] == pytest.approx(Jdot_test[0], rel=1e-4)
    assert Jdot[0, 1] == pytest.approx(Jdot_test[1], rel=1e-4)


def test_variable_map_NonZero(variable_differentialMap):
    dm_static, dm_dynamic = variable_differentialMap
    dm_static.concretize()
    dm_dynamic.concretize()
    q = np.array([1.0, -0.2])
    qdot = np.array([-0.3, 0.7])
    q_p = np.array([-0.2, 0.2])
    qdot_p = np.array([-0.4, 0.8])
    qddot_p = np.array([-0.0, 0.0])
    q_rel, qdot_rel = dm_dynamic.forward(q= q, qdot= qdot, x_ref=q_p, xdot_ref=qdot_p, xddot_ref=qddot_p)
    assert q_rel[0] == pytest.approx(q[0] - q_p[0])
    x_var, J_var, Jdot_var = dm_static.forward(q_rel=q_rel, qdot_rel=qdot_rel)
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
    assert xdot_var[0] == pytest.approx(xdot_p_test1 + xdot_p_test2, rel=1e-5)
