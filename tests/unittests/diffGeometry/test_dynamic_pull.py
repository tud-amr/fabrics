import pytest
import casadi as ca
import numpy as np
from fabrics.diffGeometry.spec import Spec
from fabrics.diffGeometry.diffMap import DifferentialMap, DynamicDifferentialMap
from fabrics.diffGeometry.geometry import Geometry

from fabrics.helpers.variables import Variables

Jdot_sign = -1

@pytest.fixture
def simple_geometry():
    x_rel = ca.SX.sym("x_rel", 1)
    xdot_rel = ca.SX.sym("xdot_rel", 1)
    var = Variables(state_variables={'x_rel': x_rel, 'xdot_rel': xdot_rel})
    h = 2 * x_rel
    return Geometry(h=h, var=var)


@pytest.fixture
def simple_spec():
    x_rel = ca.SX.sym("x_rel", 2)
    xdot_rel = ca.SX.sym("xdot_rel", 2)
    M1 = ca.SX(np.identity(2))
    f1 = -0.5 * ca.vertcat(1 / (x_rel[0] ** 2), 1 / (x_rel[1] ** 2))
    var = Variables(state_variables={'x_rel': x_rel, 'xdot_rel': xdot_rel})
    s1 = Spec(M1, f=f1, var=var)
    return s1


@pytest.fixture
def dynamic_map():
    x_ref = ca.SX.sym("x_ref", 1)
    xdot_ref = ca.SX.sym("xdot_ref", 1)
    xddot_ref = ca.SX.sym("xddot_ref", 1)
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("x_dot", 1)
    variables = Variables(state_variables={'x': x, 'xdot': xdot}, parameters={'x_ref': x_ref, 'xdot_ref': xdot_ref, 'xddot_ref': xddot_ref})
    return DynamicDifferentialMap(variables)

@pytest.fixture
def dynamic_map_2d():
    x_ref = ca.SX.sym("x_ref", 2)
    xdot_ref = ca.SX.sym("xdot_ref", 2)
    xddot_ref = ca.SX.sym("xddot_ref", 2)
    x = ca.SX.sym("x", 2)
    xdot = ca.SX.sym("x_dot", 2)
    variables = Variables(state_variables={'x': x, 'xdot': xdot}, parameters={'x_ref': x_ref, 'xdot_ref': xdot_ref, 'xddot_ref': xddot_ref})
    return DynamicDifferentialMap(variables)


def test_geometry_pullback(simple_geometry, dynamic_map):
    g = simple_geometry
    dm = dynamic_map
    # pull
    g_pulled = g.dynamic_pull(dm)
    g_pulled.concretize()
    x_np = np.array([1.2])
    xdot_np = np.array([0.1])
    x_ref_np = np.array([1.4])
    xdot_ref_np = np.array([-1.2])
    xddot_ref_np = np.array([0.2])
    h, xddot = g_pulled.evaluate(x=x_np, xdot=xdot_np, x_ref=x_ref_np, xdot_ref=xdot_ref_np, xddot_ref=xddot_ref_np)
    h_test = 2 * (x_np - x_ref_np) - xddot_ref_np
    assert h[0] == pytest.approx(h_test[0])
    assert xddot[0] == pytest.approx(-h_test[0])

def test_spec_pullback(simple_spec, dynamic_map_2d):
    s = simple_spec
    dm = dynamic_map_2d
    # pull
    s_pulled = s.dynamic_pull(dm)
    s_pulled.concretize()
    x_np = np.array([1.2, 0.1])
    xdot_np = np.array([0.1, 0.2])
    x_ref_np = np.array([1.4, 0.3])
    xdot_ref_np = np.array([-1.2, 0.1])
    xddot_ref_np = np.array([0.2, 0.0])
    M, f, xddot = s_pulled.evaluate(x=x_np, xdot=xdot_np, x_ref=x_ref_np, xdot_ref=xdot_ref_np, xddot_ref=xddot_ref_np)
    M_test = np.identity(2)
    f_test = -0.5 * np.array([1 / ((x_np[0] - x_ref_np[0]) ** 2), 1 / ((x_np[1]  - x_ref_np[1])** 2)]) - np.dot(M_test, xddot_ref_np)
    assert M[0, 0] == pytest.approx(M_test[0, 0])
    assert M[0, 1] == pytest.approx(M_test[0, 1])
    assert M[1, 0] == pytest.approx(M_test[1, 0])
    assert M[1, 1] == pytest.approx(M_test[1, 1])
    assert f[0] == pytest.approx(f_test[0])
    assert f[1] == pytest.approx(f_test[1])
