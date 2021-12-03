import pytest
import casadi as ca
import numpy as np
from optFabrics.diffGeometry.energy import Lagrangian
from optFabrics.diffGeometry.analyticSymbolicTrajectory import AnalyticSymbolicTrajectory


@pytest.fixture
def rel_lag():
    traj = ['0.5 * t']
    J = ca.SX(np.identity(1))
    refTraj = AnalyticSymbolicTrajectory(J, 1, traj=traj)
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    l_en = 0.5 * (xdot - refTraj.xdot()) ** 2
    return Lagrangian(l_en, x=x, xdot=xdot, refTrajs=[refTraj])


def test_rel_lag(rel_lag):
    ref_traj = rel_lag._refTrajs[0]
    ref_traj.concretize()
    rel_lag.concretize()
    # Evaluation
    t = 0.5
    x_p, xdot_p, xddot_p = ref_traj.evaluate(t)
    assert x_p == pytest.approx(0.25)
    assert xdot_p == pytest.approx(0.5)
    assert xddot_p == pytest.approx(0.0)
    x = np.array([0.0])
    xdot = np.array([0.0])
    M, f, H = rel_lag.evaluate(x, xdot, x_p, xdot_p, xddot_p)
    assert H == pytest.approx(0.5**3, rel=1e-4)
    assert M[0, 0] == pytest.approx(1.0, rel=1e-4)
    assert f[0] == pytest.approx(0.0, rel=1e-4)


