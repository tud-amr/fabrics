import pytest
import casadi as ca
import numpy as np
from fabrics.diffGeometry.energy import Lagrangian
from fabrics.diffGeometry.analyticSymbolicTrajectory import AnalyticSymbolicTrajectory
from fabrics.diffGeometry.diffMap import DynamicDifferentialMap

from fabrics.helpers.variables import Variables


@pytest.fixture
def rel_lag():
    traj = ['0.5 * t']
    J = ca.SX(np.identity(1))
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    q_rel = ca.SX.sym("q_rel", 1)
    qdot_rel = ca.SX.sym("qdot_rel", 1)
    q_p = ca.SX.sym("q_p", 1)
    qdot_p = ca.SX.sym("qdot_p", 1)
    qddot_p = ca.SX.sym("qddot_p", 1)
    variables = Variables(state_variables={'q_rel': q_rel, 'qdot_rel': qdot_rel})
    l_en = 0.5 * (qdot_rel) ** 2
    variables_dynamic = Variables(state_variables={'x': x, 'xdot': xdot}, parameters={'x_ref': q_p, 'xdot_ref': qdot_p, 'xddot_ref': qddot_p})
    dm = DynamicDifferentialMap(variables_dynamic)
    return Lagrangian(l_en, var=variables).dynamic_pull(dm)


def test_rel_lag(rel_lag):
    rel_lag.concretize()
    x_p = np.array([0.25])
    xdot_p = np.array([0.5])
    xddot_p = np.array([0.0])
    x = np.array([0.0])
    xdot = np.array([0.0])
    M, f, H = rel_lag.evaluate(x=x, xdot=xdot, x_ref=x_p, xdot_ref=xdot_p, xddot_ref=xddot_p)
    assert H == pytest.approx(0.5**3, rel=1e-4)
    assert M[0, 0] == pytest.approx(1.0, rel=1e-4)
    assert f[0] == pytest.approx(0.0, rel=1e-4)


