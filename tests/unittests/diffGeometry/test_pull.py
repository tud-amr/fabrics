import pytest
import casadi as ca
import numpy as np
from fabrics.diffGeometry.spec import Spec
from fabrics.diffGeometry.diffMap import DifferentialMap

Jdot_sign = -1


@pytest.fixture
def simple_spec():
    x = ca.SX.sym("x", 2)
    xdot = ca.SX.sym("xdot", 2)
    M1 = ca.SX(np.identity(2))
    f1 = -0.5 * ca.vertcat(1 / (x[0] ** 2), 1 / (x[1] ** 2))
    s1 = Spec(M1, f=f1, x=x, xdot=xdot)
    return s1


@pytest.fixture
def simple_differentialMap():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    phi = ca.vertcat(q[0] * ca.cos(q[1]), q[0] * ca.sin(q[1]))
    dm = DifferentialMap(phi, q=q, qdot=qdot, Jdot_sign=Jdot_sign)
    return dm


def test_pullback(simple_spec, simple_differentialMap):
    s = simple_spec
    dm = simple_differentialMap
    # pull
    s_pulled = s.pull(dm)
    s_pulled.concretize()
    q = np.array([1.7, -np.pi / 3])
    qdot = np.array([1.2, 1.3])
    M, f, _ = s_pulled.evaluate({'q': q, 'qdot': qdot})
    """manually computed result"""
    cost = np.cos(q[1])
    sint = np.sin(q[1])
    r = q[0]
    x1 = cost * r
    x2 = sint * r
    tdot = qdot[1]
    rdot = qdot[0]
    Jt = np.array([[cost, sint], [-r * sint, r * cost]])
    f0 = np.array([-0.5 / (x1 ** 2), -0.5 / (x2 ** 2)])
    f1 = np.dot(Jt, f0)
    JtMJdot = Jdot_sign * np.array([[0, -r * tdot], [r * tdot, r * rdot]])
    f2 = -np.dot(JtMJdot, qdot)
    """"""
    assert ca.is_equal(s_pulled.x(), dm.q())
    assert ca.is_equal(s_pulled.xdot(), dm.qdot())
    assert M[0, 0] == 1
    assert M[1, 0] == 0
    assert M[0, 1] == 0
    assert M[1, 1] == pytest.approx(q[0] ** 2)
    assert f[0] == pytest.approx(f1[0] - f2[0])
    assert f[1] == pytest.approx(f1[1] - f2[1])


def test_equal_results(simple_spec, simple_differentialMap):
    s = simple_spec
    dm = simple_differentialMap
    dm.concretize()
    s_pulled = s.pull(dm)
    s_pulled.concretize()
    s.concretize()
    q = np.array([1.7, -np.pi / 3])
    qdot = np.array([1.2, 1.3])
    x, J, Jdot = dm.forward({"q": q, "qdot": qdot})
    xdot = np.dot(J, qdot)
    Jt = np.transpose(J)
    M_q, f_q, qddot = s_pulled.evaluate({"q": q, "qdot": qdot})
    M_x, f_x, xddot = s.evaluate({"x": x, "xdot": xdot})
    xddot_man = np.dot(J, qddot) + np.dot(Jdot, qdot)
    f_q_man = np.dot(Jt, f_x) + np.dot(Jt, np.dot(M_x, np.dot(Jdot, qdot)))
    M_q_man = np.dot(Jt, np.dot(M_x, J))
    assert M_q[0, 0] == pytest.approx(M_q_man[0, 0])
    assert M_q[1, 0] == pytest.approx(M_q_man[1, 0])
    assert M_q[0, 1] == pytest.approx(M_q_man[0, 1])
    assert M_q[1, 1] == pytest.approx(M_q_man[1, 1])
    assert xddot_man[0] == pytest.approx(xddot[0], rel=1e-4)
    assert xddot_man[1] == pytest.approx(xddot[1], rel=1e-4)
    assert f_q[0] == pytest.approx(f_q_man[0])
    assert f_q[1] == pytest.approx(f_q_man[1])
