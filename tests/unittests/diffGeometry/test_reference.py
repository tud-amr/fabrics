import pytest
import casadi as ca
import numpy as np
from optFabrics.diffGeometry.referenceTrajectory import ReferenceTrajectory
from optFabrics.diffGeometry.diffMap import DifferentialMap


def test_ref_creation():
    t = ca.SX.sym("t")
    x = ca.vertcat(3 * t, -1 * t)
    J = ca.SX(np.identity(2))
    refTraj = ReferenceTrajectory(2, J, traj=x, t=t)
    refTraj.concretize()
    assert ca.is_equal(t, refTraj._t)


@pytest.fixture
def simple_trajectory():
    t = ca.SX.sym("t")
    x = ca.vertcat(3 * t, -1 * t**2)
    J = ca.SX(np.identity(2))
    refTraj = ReferenceTrajectory(2, J, traj=x, t=t)
    return refTraj


@pytest.fixture
def simple_map():
    q = ca.SX.sym("q", 3)
    qdot = ca.SX.sym("qdot", 3)
    dm = DifferentialMap(q[0:2], q=q, qdot=qdot)
    return dm


def test_ref_evaluation(simple_trajectory):
    refTraj = simple_trajectory
    refTraj.concretize()
    t = 0.3
    x, v, a = refTraj.evaluate(t)
    assert x[0] == pytest.approx(0.9)
    assert x[1] == pytest.approx(-0.09)
    assert v[0] == pytest.approx(3.0)
    assert v[1] == pytest.approx(-0.6)
    assert a[0] == pytest.approx(0.0)
    assert a[1] == pytest.approx(-2.0)


def test_pull_refTraj(simple_trajectory, simple_map):
    refTraj = simple_trajectory
    dm = simple_map
    xdot_p = refTraj.xdot()
    assert isinstance(xdot_p, ca.SX)
    assert (2, 1) == xdot_p.size()
    refTraj_pull = refTraj.pull(dm)
    xdot_pull_p = refTraj_pull.xdot()
    assert isinstance(xdot_pull_p, ca.SX)
    assert (3, 1) == xdot_pull_p.size()
