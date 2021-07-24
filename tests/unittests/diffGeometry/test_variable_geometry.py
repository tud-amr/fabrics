import pytest
import casadi as ca
import numpy as np
from optFabrics.diffGeometry.geometry import Geometry
from optFabrics.diffGeometry.spec import Spec
from optFabrics.diffGeometry.diffMap import VariableDifferentialMap


@pytest.fixture
def variable_geometry():
    q = ca.SX.sym("q", 1)
    qdot = ca.SX.sym("qdot", 1)
    q_p = ca.SX.sym("q_p", 1)
    qdot_p = ca.SX.sym("qdot_p", 1)
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    phi = ca.fabs(q-q_p)
    dm = VariableDifferentialMap(phi, q=q, qdot=qdot, q_p=q_p, qdot_p=qdot_p)
    h = 0.5 / (x ** 2) * ca.norm_2(xdot) ** 2
    geo = Geometry(h=h, x=x, xdot=xdot)
    return dm, geo


def test_variable_geometry(variable_geometry):
    dm, geo = variable_geometry
    geo_var = geo.pull(dm)
    geo_var.concretize()
    q = np.array([1.0])
    qdot = np.array([-0.2])
    q_p = np.array([0.2])
    qdot_p = np.array([1.0])
    h, qddot = geo_var.evaluate(q, qdot, q_p, qdot_p)
    h_test = 1 / (2 * np.linalg.norm(q - q_p)**2) * np.linalg.norm(qdot-qdot_p)**2
    assert isinstance(h, np.ndarray)
    assert h[0] == pytest.approx(h_test)
    assert qddot[0] == pytest.approx(-h_test)
    # must equal to summed motion for the qdot and qdot_p = 0
    qdot_pure = qdot - qdot_p
    h_pure, _ = geo_var.evaluate(q, qdot_pure, q_p, np.zeros(1))
    assert h_pure[0] == pytest.approx(h_test)
