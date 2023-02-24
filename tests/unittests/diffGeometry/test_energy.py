import pytest
import casadi as ca
import numpy as np
from fabrics.diffGeometry.spec import Spec
from fabrics.diffGeometry.energy import Lagrangian, FinslerStructure
from fabrics.diffGeometry.diffMap import DifferentialMap

from fabrics.helpers.variables import Variables


@pytest.fixture
def simple_lagrangian():
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    lam = 0.25
    l = 0.5 * lam / x * xdot ** 2
    lagrangian = Lagrangian(l, x=x, xdot=xdot)
    return lagrangian


@pytest.fixture
def simple_finsler_structure():
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    lam = 0.25
    lg = lam / x * xdot
    finslerStruct = FinslerStructure(lg, x=x, xdot=xdot)
    return finslerStruct


@pytest.fixture
def two_dimensional_lagrangian():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    x = ca.SX.sym("x", 2)
    xdot = ca.SX.sym("xdot", 2)
    lam = 0.25
    l = 0.5 * lam / (ca.norm_2(x)**2) * ca.norm_2(xdot)
    lg = Lagrangian(l, x=x, xdot=xdot)
    phi = ca.vertcat(ca.cos(q[1]) * q[0], ca.sin(q[1]) * q[0])
    variables = Variables(state_variables={'q': q, 'qdot': qdot})
    dm = DifferentialMap(phi, variables, Jdot_sign=+1)
    return lg, dm


def test_simple_lagrangian(simple_lagrangian):
    simple_lagrangian.concretize()
    x = np.array([1.2])
    xdot = np.array([2.0])
    f_man = -0.125 * xdot ** 2 / (x ** 2)
    M_man = 0.25 / x
    l_man = 0.5 * 0.25 / x * xdot ** 2
    M, f, l = simple_lagrangian.evaluate(x=x, xdot=xdot)
    assert isinstance(M, np.ndarray)
    assert isinstance(f, np.ndarray)
    assert isinstance(l, np.ndarray)
    assert l == pytest.approx(l_man)
    assert f == pytest.approx(f_man)
    assert M[0] == pytest.approx(M_man[0])


def test_simple_finsler_struct(simple_finsler_structure):
    simple_finsler_structure.concretize()
    x = np.array([1.2])
    xdot = np.array([-1.4])
    M_man = 0.25 ** 2 / (x ** 2)
    lg_man = (0.25 / x) * xdot
    l_man = 0.5 * lg_man ** 2
    f_man = -2 * 0.25 ** 2 / (x ** 3) * xdot ** 2 * (1 - 1 / 2)
    M, f, l, lg = simple_finsler_structure.evaluate(x=x, xdot=xdot)
    assert isinstance(M, np.ndarray)
    assert isinstance(f, np.ndarray)
    assert isinstance(lg, np.ndarray)
    assert isinstance(l, np.ndarray)
    assert lg == pytest.approx(lg_man[0])
    assert l == pytest.approx(l_man[0])
    assert f == pytest.approx(f_man)
    assert M[0] == pytest.approx(M_man[0])

def test_pull_lagrangian(two_dimensional_lagrangian):
    lg, dm = two_dimensional_lagrangian
    lg.concretize()
    dm.concretize()
    lg_pulled = lg.pull(dm)
    lg_pulled.concretize()
    q = np.array([1.0, -0.23])
    qdot = np.array([0.2, 0.6])
    x, J, Jdot = dm.forward(q=q, qdot=qdot)
    Jt = np.transpose(J)
    xdot = np.dot(J, qdot)
    M, f, l = lg.evaluate(x=x, xdot=xdot)
    M_p, f_p, l_p = lg_pulled.evaluate(q=q, qdot=qdot)
    M_p_test = np.dot(Jt, np.dot(M, J))
    f_p_test = np.dot(Jt, f) + np.dot(Jt, np.dot(M, np.dot(Jdot, qdot)))
    assert l == pytest.approx(l_p)
    assert M_p_test[0, 0] == pytest.approx(M_p[0, 0])
    assert M_p_test[0, 1] == pytest.approx(M_p[0, 1])
    assert M_p_test[1, 0] == pytest.approx(M_p[1, 0])
    assert M_p_test[1, 1] == pytest.approx(M_p[1, 1])
    assert f_p_test[0] == pytest.approx(f_p[0])
    assert f_p_test[1] == pytest.approx(f_p[1])

