import pytest
import casadi as ca
import numpy as np
from optFabrics.diffGeometry.spec import Spec, SpecException
from optFabrics.diffGeometry.diffMap import DifferentialMap


@pytest.fixture
def simple_spec():
    x = ca.SX.sym("x", 2)
    xdot = ca.SX.sym("xdot", 2)
    M1 = ca.SX(np.identity(2))
    f1 = -0.5 * ca.vertcat(1/ (x[0] ** 2), 1 / (x[1] ** 2))
    s1 = Spec(M1, f1, x, xdot)
    return s1

@pytest.fixture
def simple_differentialMap():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    phi = ca.vertcat(q[0] * ca.cos(q[1]), q[0] * ca.sin(q[1]))
    dm = DifferentialMap(q, qdot, phi)
    return dm

def test_pullback(simple_spec, simple_differentialMap):
    s = simple_spec
    dm = simple_differentialMap
    # pull
    s_pulled = s.pull(dm)
    s_pulled.concretize()
    q = np.array([1.7, -np.pi/3])
    qdot = np.array([1.2, 1.3])
    M, f = s_pulled.evaluate(q, qdot)
    """manually computed result"""
    cost = np.cos(q[1])
    sint = np.sin(q[1])
    r = q[0]
    x1 = cost * r
    x2 = sint * r
    tdot = qdot[1]
    rdot = qdot[0]
    Jt = np.array([[cost, sint], [-r*sint, r*cost]])
    f0 = np.array([-0.5/(x1**2), -0.5/(x2**2)])
    f1 = np.dot(Jt, f0)
    JtMJdot = np.array([
        [0, -r * tdot],
        [r * tdot, r*rdot]
        ])
    f2 = np.dot(JtMJdot, qdot)
    """"""
    assert ca.is_equal(s_pulled._x, dm._q)
    assert ca.is_equal(s_pulled._xdot, dm._qdot)
    assert M[0, 0] == 1
    assert M[1, 0] == 0
    assert M[0, 1] == 0
    assert M[1, 1] == pytest.approx(q[0]**2)
    assert f[0] == pytest.approx(f1[0] + f2[0])
    assert f[1] == pytest.approx(f1[1] + f2[1])


