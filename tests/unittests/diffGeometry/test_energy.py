import pytest
import casadi as ca
import numpy as np
from optFabrics.diffGeometry.spec import Spec, SpecException
from optFabrics.diffGeometry.energy import Lagrangian, FinslerStructure


@pytest.fixture
def simple_lagrangian():
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    lam = 0.25
    l = 0.5 * lam/x * xdot**2
    lagrangian = Lagrangian(l, x, xdot)
    return lagrangian

@pytest.fixture
def simple_finsler_structure():
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    lam = 0.25
    lg = lam/x * xdot
    finslerStruct = FinslerStructure(lg, x, xdot)
    return finslerStruct

def test_simple_lagrangian(simple_lagrangian):
    simple_lagrangian.concretize()
    x = np.array([1.2])
    xdot = np.array([2.0])
    f_man = -0.125 * xdot**2 / (x**2)
    M_man = 0.25 / x
    l_man = 0.5 * 0.25/ x * xdot**2
    M, f, l = simple_lagrangian.evaluate(x, xdot)
    assert isinstance(M, np.ndarray)
    assert isinstance(f, np.ndarray)
    assert isinstance(l, float)
    assert l == pytest.approx(l_man)
    assert f == pytest.approx(f_man)
    assert M[0, 0] == pytest.approx(M_man[0])

def test_simple_finsler_struct(simple_finsler_structure):
    simple_finsler_structure.concretize()
    x = np.array([1.2])
    xdot = np.array([-1.4])
    M_man = 0.25**2 / (x**2)
    lg_man = (0.25/ x) * xdot
    l_man = 0.5 * lg_man**2
    f_man = -2 * 0.25**2/ (x**3) * xdot**2 * (1 - 1/2)
    M, f, l, lg = simple_finsler_structure.evaluate(x, xdot)
    assert isinstance(M, np.ndarray)
    assert isinstance(f, np.ndarray)
    assert isinstance(lg, float)
    assert isinstance(l, float)
    assert lg == pytest.approx(lg_man[0])
    assert l == pytest.approx(l_man[0])
    assert f == pytest.approx(f_man)
    assert M[0, 0] == pytest.approx(M_man[0])
