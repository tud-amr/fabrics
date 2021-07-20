import pytest
import casadi as ca
import numpy as np
from optFabrics.diffGeometry.geometry import Geometry
from optFabrics.diffGeometry.energy import Lagrangian, FinslerStructure
from optFabrics.diffGeometry.diffMap import DifferentialMap
from optFabrics.diffGeometry.variables import eps


@pytest.fixture
def simple_geometry():
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    h = 0.5 / (x ** 2) * ca.norm_2(xdot)**2
    geo = Geometry(h, x, xdot)
    return geo

@pytest.fixture
def energization_example():
    x = ca.SX.sym("x", 2)
    xdot = ca.SX.sym("xdot", 2)
    h = 0.5 / (x ** 2) * ca.norm_2(xdot)**2
    geo = Geometry(h, x, xdot)
    l = 1.0 * ca.norm_2(xdot)
    le = FinslerStructure(l, x, xdot)
    return geo, le

@pytest.fixture
def energization_example_pulled():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    x = ca.SX.sym("x", 2)
    xdot = ca.SX.sym("xdot", 2)
    phi = ca.vertcat(ca.cos(q[1]) * q[0], ca.sin(q[1]) * q[0])
    dm = DifferentialMap(q, qdot, phi)
    h = 0.5 * ca.norm_2(xdot)**2/ (x ** 2 + eps)
    geo = Geometry(h, x, xdot)
    l = 0.5 * ca.dot(x, x) * ca.dot(xdot, xdot)
    le = Lagrangian(l, x, xdot)
    return geo, le, dm

def test_simple_spec(simple_geometry):
    simple_geometry.concretize()
    x = np.array([1.0])
    xdot = np.array([1.0])
    M, h, _ = simple_geometry.evaluate(x, xdot)
    assert isinstance(M, np.ndarray)
    assert isinstance(h, np.ndarray)
    assert M == 1.0
    assert h[0] == 0.5

def test_energization(energization_example):
    geo = energization_example[0]
    le = energization_example[1]
    geo_energized = geo.energize(le)
    geo_energized.concretize()
    geo_weighted = geo.addMetric(le)
    geo_weighted.concretize()
    x = np.array([0.2, -0.8])
    xdot = np.array([-0.5, -1.4])
    xdot_norm2 = xdot[0]**2 + xdot[1]**2
    pe = np.array([[1 - ((xdot[0]**2)/xdot_norm2), -xdot[0]*xdot[1]/xdot_norm2], [-xdot[0]*xdot[1]/xdot_norm2, 1 - (xdot[1]**2/xdot_norm2)]])
    h = 0.5 / (x ** 2) * np.linalg.norm(xdot)**2
    f_test = np.dot(pe, h)
    M, f, _ = geo_energized.evaluate(x, xdot)
    M_w, f_w, _, alpha_ex = geo_weighted.evaluate(x, xdot)
    h_alpha = h + alpha_ex * xdot
    assert M[0, 0] == 1.0
    assert M[0, 1] == 0.0
    assert M[1, 0] == 0.0
    assert M[1, 1] == 1.0
    assert f[0] == pytest.approx(f_test[0])
    assert f[1] == pytest.approx(f_test[1])
    assert h_alpha[0] == pytest.approx(f_test[0])
    assert h_alpha[1] == pytest.approx(f_test[1])

def test_pull_energized(energization_example_pulled):
    geo, le, dm = energization_example_pulled
    geo.concretize()
    dm.concretize()
    geo_energized = geo.energize(le)
    geo_energized.concretize()
    geo_weighted = geo.addMetric(le)
    geo_weighted.concretize()
    q = np.array([0.5, np.pi/4])
    qdot = np.array([-0.0, 5.2])
    x, J, Jdot = dm.forward(q, qdot)
    Jt = np.transpose(J)
    xdot = np.dot(J, qdot)
    M_0, f_0, xddot_0 = geo.evaluate(x, xdot)
    M, f, xddot = geo_energized.evaluate(x, xdot)
    M_w, f_w, xddot_w, alpha_w = geo_weighted.evaluate(x, xdot)
    xddot_w_alpha = xddot_w - alpha_w * xdot
    fe = 2 * np.dot(np.outer(x, xdot), xdot) - np.dot(xdot, xdot) * x
    h = 0.5 * np.linalg.norm(xdot)**2/ (x ** 2 + eps)
    xddot_w_test = -np.dot(np.dot(np.linalg.pinv(M_w), M_w), h)
    f_w_test = np.dot(M_w, h)
    assert M[0, 0] == pytest.approx(x[0]**2 + x[1]**2)
    assert M[0, 1] == pytest.approx(0.0)
    assert M[1, 0] == pytest.approx(0.0)
    assert M[1, 1] == pytest.approx(x[0]**2 + x[1]**2)
    assert M_w[0, 0] == pytest.approx(x[0]**2 + x[1]**2)
    assert M_w[0, 1] == pytest.approx(0.0)
    assert M_w[1, 0] == pytest.approx(0.0)
    assert M_w[1, 1] == pytest.approx(x[0]**2 + x[1]**2)
    assert f_w[0] == pytest.approx(f_w_test[0])
    assert f_w[1] == pytest.approx(f_w_test[1])
    assert xddot_w[0] == pytest.approx(xddot_w_test[0])
    assert xddot_w[1] == pytest.approx(xddot_w_test[1])
    assert xddot[0] == pytest.approx(xddot_w_alpha[0], abs=1e-4)
    assert xddot[1] == pytest.approx(xddot_w_alpha[1], abs=1e-4)
    geo_pulled = geo.pull(dm)
    geo_pulled.concretize()
    geo_energized_pulled = geo_energized.pull(dm)
    geo_energized_pulled.concretize()
    geo_weighted_pulled = geo_weighted.pull(dm)
    geo_weighted_pulled.concretize()
    M_p_0, f_p_0, qddot_0 = geo_pulled.evaluate(q, qdot)
    M_p, f_p, qddot = geo_energized_pulled.evaluate(q, qdot)
    M_p_w, f_p_w, qddot_w, alpha_p_w = geo_weighted_pulled.evaluate(q, qdot)
    qddot_w_alpha = qddot_w - alpha_p_w * qdot
    f_p_w_test = np.dot(Jt, np.dot(M_w, h)) + np.dot(Jt, np.dot(M_w, np.dot(Jdot, qdot)))
    M_p_test = np.dot(np.transpose(J), np.dot(M, J))
    assert M_p[0, 0] == pytest.approx(M_p_test[0, 0])
    assert M_p[0, 1] == pytest.approx(M_p_test[0, 1])
    assert M_p[1, 0] == pytest.approx(M_p_test[1, 0])
    assert M_p[1, 1] == pytest.approx(M_p_test[1, 1])
    assert M_p_w[0, 0] == pytest.approx(M_p_test[0, 0])
    assert M_p_w[0, 1] == pytest.approx(M_p_test[0, 1])
    assert M_p_w[1, 0] == pytest.approx(M_p_test[1, 0])
    assert M_p_w[1, 1] == pytest.approx(M_p_test[1, 1])
    assert f_p_w[0] == pytest.approx(f_p_w_test[0])
    assert f_p_w[1] == pytest.approx(f_p_w_test[1])
    assert qddot[0] == pytest.approx(qddot_w_alpha[0], abs=1e-6)
    assert qddot[1] == pytest.approx(qddot_w_alpha[1], abs=1e-6)

def test_homogeneous_degree2(simple_geometry):
    simple_geometry.concretize()
    isHomogeneousDegree2 = simple_geometry.testHomogeneousDegree2()
    assert isHomogeneousDegree2 == True


