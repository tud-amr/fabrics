import pytest
import casadi as ca
import numpy as np
from optFabrics.planner.nonHolonomicPlanner import NonHolonomicPlanner
from optFabrics.diffGeometry.diffMap import DifferentialMap
from optFabrics.diffGeometry.energy import FinslerStructure, Lagrangian
from optFabrics.diffGeometry.geometry import Geometry


@pytest.fixture
def nonHolonomic_planner():
    # variables x, y, theta
    x = ca.SX.sym("x", 3)
    xdot = ca.SX.sym("xdot", 3)
    qdot = ca.SX.sym("qdot", 2)
    l = 0.5 * ca.dot(xdot, xdot)
    l_base = Lagrangian(l, x=x, xdot=xdot)
    h_base = ca.SX(np.zeros(3))
    geo_base = Geometry(h=h_base, var=[x, xdot])
    l_front = 0.2
    J_nh = ca.SX(
        np.array([
            [ca.cos(x[2]), -l_front * ca.sin(x[2])],
            [ca.sin(x[2]), +l_front * ca.cos(x[2])],
            [0, 1]
        ])
    )
    f_extra = qdot[0] * qdot[1] * ca.vertcat(-ca.sin(x[2]), ca.cos(x[2]), 0)
    planner = NonHolonomicPlanner(geo_base, l_base, J_nh, qdot, f_extra)
    return planner


@pytest.fixture
def nonHolonomic_col():
    x = ca.SX.sym("x", 3)
    xdot = ca.SX.sym("xdot", 3)
    qdot = ca.SX.sym("qdot", 2)
    l = 0.5 * ca.dot(xdot[0:3], xdot[0:3])
    l_base = Lagrangian(l, x=x, xdot=xdot)
    h_base = ca.SX(np.zeros(3))
    geo_base = Geometry(h=h_base, var=[x, xdot])
    l_front = 0.2
    J_nh = ca.SX(
        np.array([
            [ca.cos(x[2]), -l_front * ca.sin(x[2])],
            [ca.sin(x[2]), +l_front * ca.cos(x[2])],
            [0, 1]
        ])
    )
    f_extra = qdot[0] * qdot[1] * ca.vertcat(-ca.sin(x[2]), ca.cos(x[2]), 0)
    planner = NonHolonomicPlanner(geo_base, l_base, J_nh, qdot, f_extra)

    x_col = ca.SX.sym("x", 1)
    xdot_col = ca.SX.sym("xdot", 1)
    l = 0.5 * ca.dot(xdot_col, xdot_col)
    x_obst = np.array([1.0, 0.0])
    phi = ca.norm_2(x[0:2] - x_obst)
    dm = DifferentialMap(phi, q=x, qdot=xdot)
    s = -0.5 * (ca.sign(xdot_col) - 1)
    lg = 1 / x_col * s * xdot_col
    l = FinslerStructure(lg, x=x_col, xdot=xdot_col)
    h = 0.5 / (x_col ** 2) * xdot_col
    geo = Geometry(h=h, x=x_col, xdot=xdot_col)
    return planner, dm, l, geo


@pytest.mark.skip(reason="Nonholonomic base currently not maintained")
def test_nh_planner_zero(nonHolonomic_planner):
    nh_planner = nonHolonomic_planner
    nh_planner.concretize()
    x = np.array([0.0, 0.0, 0.0])
    xdot = np.array([0.0, 0.0, 0.0])
    qdot = np.zeros(2)
    action = nh_planner.computeAction(x, xdot, qdot)
    assert len(action) == 2
    assert action[0] == pytest.approx(0.0)
    assert action[1] == pytest.approx(0.0)


@pytest.mark.skip(reason="Nonholonomic base currently not maintained")
def test_nh_planner_col(nonHolonomic_col):
    planner, dm, l, geo = nonHolonomic_col
    dm.concretize()
    geo.concretize()
    l.concretize()
    planner.addGeometry(dm, l, geo)
    planner.concretize()
    eg = planner._eg
    eg.concretize()
    x = np.array([0.0, 0.0, np.pi])
    qdot = np.array([2.0, 0.0])
    xdot = np.array([np.cos(x[2]) * qdot[0], np.sin(x[2]) * qdot[0], qdot[1]])
    action = planner.computeAction(x, xdot, qdot)
    assert len(action) == 2
    assert action[0] == pytest.approx(0.0)
    assert action[1] == pytest.approx(0.0)
    x = np.array([0.0, 0.0, 0.0])
    qdot = np.array([2.0, 0.0])
    xdot = np.array([np.cos(x[2]) * qdot[0], np.sin(x[2]) * qdot[0], qdot[1]])
    action = planner.computeAction(x, xdot, qdot)
    assert len(action) == 2
    assert action[0] == pytest.approx(-0.5)
    assert action[1] == pytest.approx(0.0)
    x = np.array([0.0, 0.0, np.pi / 2 + 0.0001])
    qdot = np.array([2.0, 0.0])
    xdot = np.array([np.cos(x[2]) * qdot[0], np.sin(x[2]) * qdot[0], qdot[1]])
    action = planner.computeAction(x, xdot, qdot)
    assert len(action) == 2
    assert action[0] == pytest.approx(-0.0)
    assert action[1] == pytest.approx(0.0)
    x = np.array([0.0, 0.1, 0.0])
    qdot = np.array([2.0, 0.0])
    xdot = np.array([np.cos(x[2]) * qdot[0], np.sin(x[2]) * qdot[0], qdot[1]])
    action = planner.computeAction(x, xdot, qdot)
    assert len(action) == 2
    assert action[0] < 0.0
    assert action[1] > 0.0
    x = np.array([0.0, -0.1, 0.0])
    qdot = np.array([2.0, 0.0])
    xdot = np.array([np.cos(x[2]) * qdot[0], np.sin(x[2]) * qdot[0], qdot[1]])
    action = planner.computeAction(x, xdot, qdot)
    assert len(action) == 2
    assert action[0] < 0.0
    assert action[1] < 0.0
    x = np.array([0.0, 0.1, 0.0])
    qdot = np.array([2.0, 1.0])
    xdot = np.array([np.cos(x[2]) * qdot[0], np.sin(x[2]) * qdot[0], qdot[1]])
    action = planner.computeAction(x, xdot, qdot)
    assert len(action) == 2
    assert action[0] < 0.0
    assert action[1] > 0.0
