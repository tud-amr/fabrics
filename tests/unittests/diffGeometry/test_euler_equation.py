import pdb
import pytest
import casadi as ca
import numpy as np

from fabrics.diffGeometry.energy import Lagrangian
from fabrics.diffGeometry.diffMap import DifferentialMap, DynamicDifferentialMap

from fabrics.helpers.variables import Variables


@pytest.fixture
def relative_lagrangian():
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    x_p = ca.SX.sym("x_p", 1)
    xdot_p = ca.SX.sym("xdot_p", 1)
    xddot_p = ca.SX.sym("xddot_p", 1)
    x_rel = ca.SX.sym('x_rel', 1)
    xdot_rel = ca.SX.sym('xdot_rel', 1)
    l_rel = 0.5 * ca.dot(xdot - xdot_p, xdot - xdot_p) \
                * ca.dot(x - x_p, x - x_p)
    variables = Variables(state_variables={'x': x, 'xdot': xdot}, parameters={'x_ref': x_p, 'xdot_ref': xdot_p, 'xddot_ref': xddot_p})
    variables_relative = Variables(state_variables={'x_rel': x_rel, 'xdot_rel': xdot_rel})
    lag = Lagrangian(
                l_rel,
                var=variables_relative
        )
    dm_dynamic = DynamicDifferentialMap(variables)
    return lag.dynamic_pull(dm_dynamic)


@pytest.fixture
def static_map():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    phi = ca.norm_2(q)
    variables = Variables(state_variables={'x': q, 'xdot': qdot})
    dm = DifferentialMap(phi, variables, Jdot_sign=+1)
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
    M_lag, f_lag, h_lag = lag.evaluate(x=x, xdot=xdot, x_ref=x_p, xdot_ref=xdot_p, xddot_ref=xddot_p)
    assert isinstance(M_lag, np.ndarray)
    assert isinstance(f_lag, np.ndarray)
    assert isinstance(h_lag, np.ndarray)
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
    x, J, Jdot = dm.forward(x=q, xdot=qdot)
    xdot = np.dot(J, qdot)
    Jt = np.transpose(J)
    Jinv = np.dot(np.linalg.pinv(np.dot(Jt, J)), Jt)
    M_leaf, f_leaf, h_leaf = lag.evaluate(x=x, xdot=xdot, x_ref=x_p, xdot_ref=xdot_p, xddot_ref=xddot_p)
    M_root, f_root, h_root = lag_pull.evaluate(x=q, xdot=qdot, x_ref=x_p, xdot_ref=xdot_p, xddot_ref=xddot_p)
    M_test = np.dot(Jt, np.dot(M_leaf, J))
    f_test = np.dot(Jt, f_leaf) + np.dot(Jt, np.dot(M_leaf, np.dot(Jdot, qdot)))
    assert isinstance(h_root, np.ndarray)
    assert M_root.shape == (2, 2)
    assert f_root.shape == (2, )
    assert h_root == pytest.approx(h_leaf)
    assert M_test == pytest.approx(M_root)
    assert f_test == pytest.approx(f_root)
    xdot_comb = lag_pull.xdot_rel()
    xdot_comb_fun = ca.Function(
                'xdot_comb',
                [
                    lag_pull.x(),
                    lag_pull.xdot(), 
                    lag_pull._vars.parameters()['xdot_ref'],
                ],
                [xdot_comb]
            )
    xdot_comb_t = np.array(xdot_comb_fun(q, qdot, xdot_p))[:, 0]
    xdot_comb_test = qdot - np.dot(Jinv, xdot_p)
    assert xdot_comb_t == pytest.approx(xdot_comb_test, rel=1e-3)

