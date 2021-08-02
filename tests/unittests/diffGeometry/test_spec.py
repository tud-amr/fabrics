import pytest
import casadi as ca
import numpy as np
from optFabrics.diffGeometry.spec import Spec, SpecException


@pytest.fixture
def simple_spec():
    x = ca.SX.sym("x", 2)
    xdot = ca.SX.sym("xdot", 2)
    M1 = ca.SX(np.identity(2))
    f1 = -0.5 / (x ** 2)
    s1 = Spec(M1, f1, x=x, xdot=xdot)
    M2 = ca.SX(np.identity(2) * 0.5)
    f2 = -2.5 / (x ** 2)
    s2 = Spec(M2, f2, x=x, xdot=xdot)
    return s1, s2


@pytest.fixture
def second_spec():
    x = ca.SX.sym("x", 3)
    xdot = ca.SX.sym("xdot", 3)
    M1 = ca.SX(np.identity(3))
    f1 = -0.5 / (x ** 2)
    s1 = Spec(M1, f1, x=x, xdot=xdot)
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    M2 = ca.SX(np.identity(2))
    f2 = -0.5 / (q ** 2)
    s2 = Spec(M2, f2, x=q, xdot=qdot)
    return s1, s2


@pytest.fixture
def third_spec():
    x = np.zeros(3)
    xdot = ca.SX.sym("xdot", 3)
    M1 = ca.SX(np.identity(3))
    f1 = -0.5 / (x ** 2)
    s1 = Spec(M1, f1, var=[x, xdot])
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    M2 = np.identity(2)
    f2 = -0.5 / (q ** 2)
    s2 = Spec(M2, f2, var=[q, qdot])
    return s1, s2

@pytest.fixture
def var_spec():
    x = ca.SX.sym("x", 2)
    xdot = ca.SX.sym("xdot", 2)
    x1 = ca.SX.sym('x1', 2)
    x1dot = ca.SX.sym("x1dot", 2)
    x2 = ca.SX.sym('x2', 2)
    x2dot = ca.SX.sym("x2dot", 2)
    M1 = ca.SX(np.identity(2))
    f1 = -0.5 / (x ** 2)
    s1 = Spec(M1, f1, var=[x, xdot, x1, x1dot])
    s2 = Spec(M1, f1, var=[x, xdot, x2, x2dot])
    return s1, s2


def test_simple_spec(simple_spec):
    simple_spec[0].concretize()
    x = np.array([1.0, 0.5])
    xdot = np.array([1.0, 0.0])
    M, f, _ = simple_spec[0].evaluate(x, xdot)
    assert isinstance(M, np.ndarray)
    assert isinstance(f, np.ndarray)
    assert M[0, 0] == 1.0
    assert M[0, 1] == 0.0
    assert M[1, 1] == 1.0
    assert M[1, 0] == 0.0
    assert f[0] == -0.5
    assert f[1] == -2.0


def test_add_specs(simple_spec):
    s = simple_spec[0] + simple_spec[1]
    s.concretize()
    x = np.array([1.0, 0.5])
    xdot = np.array([1.0, 0.0])
    M, f, _ = s.evaluate(x, xdot)
    assert M[0, 0] == 1.5
    assert M[0, 1] == 0.0
    assert M[1, 1] == 1.5
    assert M[1, 0] == 0.0
    assert f[0] == -3.0
    assert f[1] == -12.0


def test_assertion_error_creation():
    x = ca.SX.sym("x", 2)
    xdot = np.array([1.0, 0.0])
    M1 = ca.SX(np.identity(2))
    f1 = -0.5 / (x ** 2)
    with pytest.raises(AssertionError):
        s1 = Spec(M1, f1, x=x, xdot=xdot)


def test_assertion_erros(simple_spec):
    s = simple_spec[0]
    with pytest.raises(AssertionError):
        b = s + 3.0


def test_add_wrong_specs(simple_spec, second_spec):
    with pytest.raises(SpecException):
        s = simple_spec[0] + second_spec[0]
    with pytest.raises(SpecException):
        s = simple_spec[0] + second_spec[1]


def test_join_variables(var_spec):
    s1, s2 = var_spec
    s_joint = s1 + s2
    var = s_joint._vars
    assert len(var) == 6
