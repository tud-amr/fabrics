import pytest
import casadi as ca
import numpy as np

from fabrics.diffGeometry.energy import Lagrangian
from fabrics.diffGeometry.diffMap import DifferentialMap
from fabrics.diffGeometry.analyticSymbolicTrajectory import AnalyticSymbolicTrajectory


@pytest.fixture
def relative_lagrangian():
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    x_p = ca.SX.sym("x_p", 1)
    xdot_p = ca.SX.sym("xdot_p", 1)
    xddot_p = ca.SX.sym("xddot_p", 1)
    l_rel = 0.5 * ca.dot(xdot - xdot_p, xdot - xdot_p) \
                * ca.dot(x - x_p, x - x_p)
    refTraj = AnalyticSymbolicTrajectory(ca.SX(np.identity(1)), 1, var=[x_p, xdot_p, xddot_p])
    lag = Lagrangian(
                l_rel,
                x=x, xdot=xdot,
                refTrajs=[refTraj]
        )
    return lag


@pytest.fixture
def static_map():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    phi = ca.norm_2(q)
    dm = DifferentialMap(phi, q=q, qdot=qdot)
    return dm


def test_relative_lagrangian(relative_lagrangian):
    lag = relative_lagrangian
    lag.concretize()
    x = np.array([1.2])
    xdot = np.array([2.0])
    x_p = np.array([2.0])
    xdot_p = np.array([2.0])
    xddot_p = np.array([2.0])
    x_rel = x - x_p
    xdot_rel = xdot - xdot_p
    M_man = x_rel**2
    f_man = xdot_rel**2 * x_rel - np.dot(M_man, xddot_p)
    h_man = 0.5 * xdot_rel**2 * x_rel**2
    M_lag, f_lag, h_lag = lag.evaluate(x, xdot, x_p, xdot_p, xddot_p)
    assert isinstance(M_lag, np.ndarray)
    assert isinstance(f_lag, np.ndarray)
    assert isinstance(h_lag, float)
    assert h_lag == pytest.approx(h_man)
    assert f_lag == pytest.approx(f_man)
    assert M_lag[0, 0] == pytest.approx(M_man[0])

def test_pull_lagrangian(relative_lagrangian, static_map):
    dm = static_map
    dm.concretize()
    lag = relative_lagrangian
    lag.concretize()
    lag_pull = lag.pull(dm)
    lag_pull.concretize()
    q = np.array([1.2, 0.2])
    qdot = np.array([2.0, 1.0])
    x_p = np.array([1.2])
    xdot_p = np.array([1.3])
    xddot_p = np.array([0.2])
    x, J, Jdot = dm.forward(q, qdot)
    xdot = np.dot(J, qdot)
    Jt = np.transpose(J)
    Jinv = np.dot(np.linalg.pinv(np.dot(Jt, J)), Jt)
    M_leaf, f_leaf, h_leaf = lag.evaluate(x, xdot, x_p, xdot_p, xddot_p)
    M_root, f_root, h_root = lag_pull.evaluate(q, qdot, x_p, xdot_p, xddot_p)
    M_test = np.dot(Jt, np.dot(M_leaf, J))
    f_test = np.dot(Jt, f_leaf) + np.dot(Jt, np.dot(M_leaf, np.dot(Jdot, qdot)))
    assert isinstance(h_root, float)
    assert M_root.shape == (2, 2)
    assert f_root.shape == (2, )
    assert h_root == pytest.approx(h_leaf)
    assert M_test == pytest.approx(M_root)
    assert f_test == pytest.approx(f_root)
    xdot_comb = lag_pull.xdot_rel()
    xdot_comb_fun = ca.Function('xdot_comb', [lag_pull.x(), lag_pull.xdot(), lag_pull._refTrajs[0]._vars[1]], [xdot_comb])
    xdot_comb_t = np.array(xdot_comb_fun(q, qdot, xdot_p))[:, 0]
    xdot_comb_test = qdot - np.dot(Jinv, xdot_p)
    assert xdot_comb_t == pytest.approx(xdot_comb_test, rel=1e-3)

