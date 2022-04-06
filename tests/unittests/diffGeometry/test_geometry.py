import pytest
import casadi as ca
import numpy as np
from fabrics.diffGeometry.geometry import Geometry
from fabrics.diffGeometry.spec import Spec

from fabrics.helpers.variables import Variables


@pytest.fixture
def simple_geometry():
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    h = 0.5 / (x ** 2) * ca.norm_2(xdot) ** 2
    geo = Geometry(h=h, x=x, xdot=xdot)
    return geo


@pytest.fixture
def simple_spec():
    x = ca.SX.sym("x", 2)
    xdot = ca.SX.sym("xdot", 2)
    M1 = ca.SX(np.array([[-0.3, 0.5], [0.5, 1.2]]))
    f1 = -0.5 / (x ** 2)
    s1_vars = Variables(state_variables={'x': x, 'xdot': xdot})
    s1 = Spec(M1, f=f1, var=s1_vars)
    return s1


def test_simple_geometry(simple_geometry):
    simple_geometry.concretize()
    x = np.array([1.0])
    xdot = np.array([1.0])
    h, xddot = simple_geometry.evaluate(x=x, xdot=xdot)
    assert isinstance(h, np.ndarray)
    assert h[0] == 0.5
    assert xddot[0] == -0.5


def test_spec2geometry(simple_spec):
    s = simple_spec
    g = Geometry(s=s)
    s.concretize()
    g.concretize()
    x = np.array([1.0, -0.3])
    xdot = np.array([-0.3, 1.2])
    _, xddot_g = g.evaluate(x=x, xdot=xdot)
    _, _, xddot_s = s.evaluate(x=x, xdot=xdot)
    assert xddot_g[0] == pytest.approx(xddot_s[0])
    assert xddot_g[1] == pytest.approx(xddot_s[1])


def test_homogeneous_degree2(simple_geometry):
    simple_geometry.concretize()
    isHomogeneousDegree2 = simple_geometry.testHomogeneousDegree2()
    assert isHomogeneousDegree2
