import pytest
import casadi as ca
import numpy as np
from fabrics.diffGeometry.geometry import Geometry
from fabrics.diffGeometry.spec import Spec
from fabrics.diffGeometry.diffMap import DifferentialMap, RelativeDifferentialMap
from fabrics.diffGeometry.analyticSymbolicTrajectory import AnalyticSymbolicTrajectory


from fabrics.helpers.variables import Variables

@pytest.fixture
def variable_geometry():
    q = ca.SX.sym("q", 1)
    qdot = ca.SX.sym("qdot", 1)
    q_p = ca.SX.sym("q_p", 1)
    qdot_p = ca.SX.sym("qdot_p", 1)
    qddot_p = ca.SX.sym("qddot_p", 1)
    variables = Variables(parameters= {'q_p': q_p, 'qdot_p': qdot_p, 'qddot_p': qddot_p})
    refTraj = AnalyticSymbolicTrajectory(ca.SX(np.identity(1)), 1, var=variables)
    q_rel = ca.SX.sym("q_rel", 1)
    qdot_rel = ca.SX.sym("qdot_rel", 1)
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    dm_rel = RelativeDifferentialMap(q=q, qdot=qdot, refTraj=refTraj)
    dm = DifferentialMap(ca.fabs(q_rel), q=q_rel, qdot=qdot_rel)
    h = 0.5 / (x ** 2) * ca.norm_2(xdot) ** 2
    geo = Geometry(h=h, x=x, xdot=xdot)
    return dm_rel, dm, geo


@pytest.fixture
def variable_spec():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    q_p = ca.SX.sym("q_p", 2)
    qdot_p = ca.SX.sym("qdot_p", 2)
    qddot_p = ca.SX.sym("qddot_p", 2)
    variables = Variables(parameters= {'q_p': q_p, 'qdot_p': qdot_p, 'qddot_p': qddot_p})
    refTraj = AnalyticSymbolicTrajectory(ca.SX(np.identity(2)), 2, var=variables)
    q_rel = ca.SX.sym("q_rel", 2)
    qdot_rel = ca.SX.sym("qdot_rel", 2)
    dm_rel = RelativeDifferentialMap(q=q, qdot=qdot, refTraj=refTraj)
    dm = DifferentialMap(ca.fabs(q_rel), q=q_rel, qdot=qdot_rel)
    x = ca.SX.sym("x", 2)
    xdot = ca.SX.sym("xdot", 2)
    M1 = ca.SX(np.identity(2))
    f1 = -0.5 / (x ** 2)
    s1 = Spec(M1, f=f1, x=x, xdot=xdot)
    return dm_rel, dm, s1


def test_variable_geometry(variable_geometry):
    dm_rel, dm, geo = variable_geometry
    geo_var = geo.pull(dm_rel).pull(dm)
    geo_var.concretize()
    q = np.array([1.0])
    qdot = np.array([-0.2])
    q_p = np.array([0.2])
    qdot_p = np.array([1.0])
    qddot_p = np.array([0.0])
    h_test = 1 / (2 * np.linalg.norm(q - q_p)**2) * np.linalg.norm(qdot-qdot_p)**2
    h, qddot = geo_var.evaluate(q=q, qdot=qdot, q_p=q_p, qdot_p=qdot_p, qddot_p=qddot_p)
    assert isinstance(h, np.ndarray)
    assert h[0] == pytest.approx(h_test, rel=1e-4)
    assert qddot[0] == pytest.approx(-h_test, rel=1e-4)
    # must equal to summed motion for the qdot and qdot_p = 0
    qdot_pure = qdot - qdot_p
    h_pure, _ = geo_var.evaluate(q=q, qdot=qdot_pure, q_p=q_p, qdot_p=np.zeros(1), qddot_p=np.zeros(1))
    assert h_pure[0] == pytest.approx(h_test, rel=1e-4)


def test_variable_spec(variable_spec):
    dm_rel, dm, s = variable_spec
    s_var = s.pull(dm).pull(dm_rel)
    s_var.concretize()
    q = np.array([1.0, 0.5])
    qdot = np.array([-0.2, 0.2])
    q_p = np.array([0.2, 0.0])
    qdot_p = np.array([1.0, 0.0])
    qddot_p = np.array([0.0, 0.0])
    M, f, xddot = s_var.evaluate(q=q, qdot=qdot, q_p=q_p, qdot_p=qdot_p, qddot_p=qddot_p)
    f_test = -0.5 / ((q-q_p) ** 2)
    M_test = np.identity(2)
    xddot_test = np.linalg.solve(M_test, -f_test)
    assert isinstance(f, np.ndarray)
    assert M_test[0, 0] == pytest.approx(M[0, 0])
    assert M_test[0, 1] == pytest.approx(M[0, 1])
    assert M_test[1, 0] == pytest.approx(M[1, 0])
    assert M_test[1, 1] == pytest.approx(M[1, 1])
    assert f[0] == pytest.approx(f_test[0])
    assert xddot[0] == pytest.approx(xddot_test[0])
