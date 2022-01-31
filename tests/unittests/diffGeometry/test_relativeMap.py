import pytest
import casadi as ca
import numpy as np
from fabrics.diffGeometry.geometry import Geometry
from fabrics.diffGeometry.spec import Spec
from fabrics.diffGeometry.energy import Lagrangian
from fabrics.diffGeometry.energized_geometry import WeightedGeometry
from fabrics.diffGeometry.diffMap import RelativeDifferentialMap
from fabrics.diffGeometry.analyticSymbolicTrajectory import AnalyticSymbolicTrajectory


@pytest.fixture
def relative_map():
    q = ca.SX.sym("q", 1)
    qdot = ca.SX.sym("qdot", 1)
    refTraj = AnalyticSymbolicTrajectory(ca.SX(np.identity(1)), 1)
    dm_rel = RelativeDifferentialMap(q=q, qdot=qdot, refTraj=refTraj)
    q_rel = ca.SX.sym("q_rel", 1)
    qdot_rel = ca.SX.sym("qdot_rel", 1)
    l_rel = 0.5 * ca.dot(qdot_rel, qdot_rel) * ca.dot(q_rel, q_rel)
    lag_rel = Lagrangian(l_rel, x=q_rel, xdot=qdot_rel)
    return dm_rel, lag_rel

@pytest.fixture
def relative_weighted_geometry(relative_map):
    dm, lag_rel = relative_map
    q = dm.q()
    qdot = dm.qdot()
    h_geo = 1/(q**2) * (qdot**2)
    geo = Geometry(h=h_geo, x=q, xdot=qdot)
    return dm, lag_rel, geo

def test_relative_lagrangian(relative_map):
    dm_rel, lag_rel = relative_map
    lag_rel.concretize()
    dm_rel.concretize()
    lag_pull = lag_rel.pull(dm_rel)
    lag_pull.concretize()
    # testing
    q = np.array([0.4])
    qdot = np.array([0.4])
    q_p = np.array([1.0])
    qdot_p = np.array([1.2])
    qddot_p = np.array([0.2])
    q_rel, qdot_rel = dm_rel.forward(q, qdot, q_p, qdot_p, qddot_p)
    M_rel, f_rel, l_rel = lag_rel.evaluate(q_rel, qdot_rel)
    M, f, l = lag_pull.evaluate(q, qdot, q_p, qdot_p, qddot_p)
    f_test = f_rel - np.dot(M_rel, qddot_p)
    assert M_rel == pytest.approx(M)
    assert f == pytest.approx(f_test)

def test_weighted_geometry(relative_weighted_geometry):
    dm_rel, lag_rel, geo = relative_weighted_geometry
    lag_pull = lag_rel.pull(dm_rel)
    eg = WeightedGeometry(g=geo,le=lag_pull) 
    dm_rel.concretize()
    lag_rel.concretize()
    lag_pull.concretize()
    geo.concretize()
    eg.concretize()
    q = np.array([0.1])
    qdot = np.array([-0.4])
    q_p = np.array([1.0])
    qdot_p = np.array([-1.2])
    qddot_p = np.array([0.7])
    q_rel, qdot_rel = dm_rel.forward(q, qdot, q_p, qdot_p, qddot_p)
    M_rel, f_rel, _ = lag_rel.evaluate(q_rel, qdot_rel)
    h, _ = geo.evaluate(q, qdot)
    M_pull, f_pull, _ = lag_pull.evaluate(q, qdot, q_p, qdot_p, qddot_p)
    assert M_rel[0, 0] == pytest.approx(M_pull[0, 0])
    frac_test = -1/(np.dot(qdot_rel, np.dot(M_rel, qdot_rel)))
    alpha_test = frac_test * np.dot(qdot_rel, np.dot(M_rel, h + qddot_p) - f_rel)
    M_eg, f_eg, l_eg, alpha_eg = eg.evaluate(q, qdot, q_p, qdot_p, qddot_p)
    assert alpha_eg == pytest.approx(alpha_test)
