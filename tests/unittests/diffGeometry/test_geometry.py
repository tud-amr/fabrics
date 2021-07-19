import pytest
import casadi as ca
import numpy as np
from optFabrics.diffGeometry.geometry import Geometry
from optFabrics.diffGeometry.energy import Lagrangian, FinslerStructure
from optFabrics.diffGeometry.diffMap import DifferentialMap


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
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    q0 = np.array([0.0, 0.0])
    phi = ca.norm_2(q - q0)
    dm = DifferentialMap(q, qdot, phi)
    h = 0.5 / (x ** 2) * ca.norm_2(xdot)**2
    geo = Geometry(h, x, xdot)
    l = 1.0 * ca.norm_2(xdot)
    le = FinslerStructure(l, x, xdot)
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
    x = np.array([0.2, -0.8])
    xdot = np.array([-0.5, -1.4])
    xdot_norm2 = xdot[0]**2 + xdot[1]**2
    pe = np.array([[1 - ((xdot[0]**2)/xdot_norm2), -xdot[0]*xdot[1]/xdot_norm2], [-xdot[0]*xdot[1]/xdot_norm2, 1 - (xdot[1]**2/xdot_norm2)]])
    h = 0.5 / (x ** 2) * np.linalg.norm(xdot)**2
    f_test = np.dot(pe, h)
    M, f, _, alpha_ex = geo_energized.evaluate(x, xdot)
    h_alpha = h + alpha_ex * xdot
    assert M[0, 0] == 1.0
    assert M[0, 1] == 0.0
    assert M[1, 0] == 0.0
    assert M[1, 1] == 1.0
    assert f[0] == pytest.approx(f_test[0])
    assert f[1] == pytest.approx(f_test[1])
    assert h_alpha[0] == pytest.approx(f_test[0])
    assert h_alpha[1] == pytest.approx(f_test[1])

def test_pulled_energization(energization_example_pulled):
    geo = energization_example_pulled[0]
    le = energization_example_pulled[1]
    dm = energization_example_pulled[2]
    dm.concretize()
    geo_energized = geo.energize(le)
    geo_energized.concretize()
    q = np.array([0.2, -0.8])
    qdot = np.array([-0.5, -1.4])
    # in task space
    x, J, _ = dm.forward(q, qdot)
    xdot = np.dot(J, qdot)
    xdot_norm2 = xdot[0]**2
    pe = 1 - (xdot[0]**2)/xdot_norm2
    h = 0.5 / (x ** 2) * np.linalg.norm(xdot)**2
    f_test = np.dot(pe, h)
    M, f, _, alpha_ex = geo_energized.evaluate(x, xdot)
    h_alpha = h + alpha_ex * xdot
    assert M[0, 0] == 1.0
    assert f[0] == pytest.approx(f_test[0], abs=1e-6)
    assert h_alpha[0] == pytest.approx(f_test[0], abs=1e-6)
    # in root space
    geo_pulled = geo_energized.pull(dm)
    geo_pulled.concretize()
    M_pulled, f_pulled, _, alpha_ex_pulled = geo_pulled.evaluate(q, qdot)
    assert alpha_ex == pytest.approx(alpha_ex_pulled)

def test_homogeneous_degree2(simple_geometry):
    simple_geometry.concretize()
    isHomogeneousDegree2 = simple_geometry.testHomogeneousDegree2()
    assert isHomogeneousDegree2 == True


if __name__ == "__main__":
    print("hi")
    e = energization_example()
    geo = e[0]
    le = e[1]
    geo_energized = geo.energize(le)

