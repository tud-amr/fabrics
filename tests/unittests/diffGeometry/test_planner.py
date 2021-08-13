import pytest
import casadi as ca
import numpy as np
from optFabrics.planner.fabricPlanner import FabricPlanner
from optFabrics.diffGeometry.diffMap import DifferentialMap, VariableDifferentialMap, RelativeDifferentialMap
from optFabrics.diffGeometry.energy import FinslerStructure, Lagrangian
from optFabrics.diffGeometry.geometry import Geometry


@pytest.fixture
def simple_planner():
    var = [ca.SX.sym("x", 1), ca.SX.sym("xdot", 1)]
    l = 0.5 * var[1]**2
    l_base = Lagrangian(l, var=var)
    geo_base = Geometry(h=ca.SX(0), var=var)
    planner = FabricPlanner(geo_base, l_base)
    return planner


@pytest.fixture
def simple_task():
    q = ca.SX.sym("q", 1)
    qdot = ca.SX.sym("qdot", 1)
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    l = 0.5 * ca.dot(qdot, qdot)
    l_base = Lagrangian(l, x=q, xdot=qdot)
    geo_base = Geometry(h=ca.SX(0), x=q, xdot=qdot)
    planner = FabricPlanner(geo_base, l_base)
    phi = ca.fabs(q - 1)
    dm = DifferentialMap(phi, q=q, qdot=qdot)
    s = -0.5 * (ca.sign(xdot) - 1)
    lg = 1 / x * s * xdot
    l = FinslerStructure(lg, x=x, xdot=xdot)
    h = 0.5 / (x ** 2) * xdot
    geo = Geometry(h=h, x=x, xdot=xdot)
    return planner, dm, l, geo


@pytest.fixture
def simple_2dtask():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    l = 0.5 * ca.dot(qdot, qdot)
    l_base = Lagrangian(l, x=q, xdot=qdot)
    geo_base = Geometry(h=ca.SX(np.zeros(2)), x=q, xdot=qdot)
    planner = FabricPlanner(geo_base, l_base)
    q0 = np.array([1.0, 0.0])
    phi = ca.norm_2(q - q0)
    dm = DifferentialMap(phi, q=q, qdot=qdot)
    s = -0.5 * (ca.sign(xdot) - 1)
    lg = 1 / x * s * xdot
    l = FinslerStructure(lg, x=x, xdot=xdot)
    h = 0.5 / (x ** 2) * xdot
    geo = Geometry(h=h, x=x, xdot=xdot)
    return planner, dm, l, geo


@pytest.fixture
def variable_2dtask():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    q_rel = ca.SX.sym("q_rel", 2)
    qdot_rel = ca.SX.sym("qdot_rel", 2)
    q_p = ca.SX.sym("q_p", 2)
    qdot_p = ca.SX.sym("qdot_p", 2)
    qddot_p = ca.SX.sym("qddot_p", 2)
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    l = 0.5 * ca.dot(qdot, qdot)
    l_base = Lagrangian(l, x=q, xdot=qdot)
    geo_base = Geometry(h=ca.SX(np.zeros(2)), x=q, xdot=qdot)
    planner = FabricPlanner(geo_base, l_base)
    dm_rel = RelativeDifferentialMap(var = [q, qdot, q_p, qdot_p, qddot_p])
    phi = ca.norm_2(q_rel)
    dm = DifferentialMap(phi, var=[q_rel, qdot_rel])
    s = -0.5 * (ca.sign(xdot) - 1)
    lg = 1 / x * s * xdot
    l = FinslerStructure(lg, x=x, xdot=xdot)
    h = 0.5 / (x ** 2) * xdot
    geo = Geometry(h=h, x=x, xdot=xdot)
    return planner, dm, dm_rel, l, geo


def test_simple_planner(simple_planner):
    simple_planner.concretize()
    x = np.array([1.0])
    xdot = np.array([1.0])
    xddot = simple_planner.computeAction(x, xdot)
    assert isinstance(xddot, np.ndarray)
    assert xddot[0] == 0.0


def test_simple_task(simple_task):
    planner = simple_task[0]
    dm = simple_task[1]
    l = simple_task[2]
    geo = simple_task[3]
    planner.addGeometry(dm, l, geo)
    planner.concretize()
    q = np.array([0.0])
    qdot = np.array([2.0])
    qddot = planner.computeAction(q, qdot)
    assert isinstance(qddot, np.ndarray)
    assert qddot[0] == pytest.approx(-2.0)


def test_simple2d_task(simple_2dtask):
    # obstacle at [1, 0]
    planner = simple_2dtask[0]
    dm = simple_2dtask[1]
    l = simple_2dtask[2]
    geo = simple_2dtask[3]
    planner.addGeometry(dm, l, geo)
    planner.concretize()
    # towards obstacle from [1, 1] with [0.0, -2.0] -> accelerate in positive y
    q = np.array([1.0, 1.0])
    qdot = np.array([0.0, -2.0])
    qddot = planner.computeAction(q, qdot)
    assert isinstance(qddot, np.ndarray)
    assert qddot[0] == pytest.approx(-0.0)
    assert qddot[1] == pytest.approx(2.0)
    # towards obstacle from [0, 0] with [2.0, 0.0] -> accelerate in negative x
    q = np.array([0.0, 0.0])
    qdot = np.array([2.0, -0.0])
    qddot = planner.computeAction(q, qdot)
    assert isinstance(qddot, np.ndarray)
    assert qddot[0] == pytest.approx(-2.0)
    assert qddot[1] == pytest.approx(0.0)
    # away from obstacle from [0, 0] with [-2.0, 0.0] -> no action
    q = np.array([0.0, 0.0])
    qdot = np.array([-2.0, -0.0])
    qddot = planner.computeAction(q, qdot)
    assert isinstance(qddot, np.ndarray)
    assert qddot[0] == pytest.approx(0.0)
    assert qddot[1] == pytest.approx(0.0)

def test_variable2d_task(variable_2dtask):
    # obstacle at [1, 0]
    planner, dm, dm_rel, l, geo = variable_2dtask
    geo_rel = geo.pull(dm)
    l_rel = l.pull(dm)
    planner.addGeometry(dm_rel, l_rel, geo_rel)
    planner.concretize()
    # towards obstacle from [1, 1] with [0.0, -2.0] -> accelerate in positive y
    q = np.array([1.0, 1.0])
    qdot = np.array([0.0, -2.0])
    q_p = np.array([1.0, 0.0])
    qdot_p = np.array([0.0, 0.0])
    qddot_p = np.array([0.0, 0.0])
    qddot = planner.computeAction(q, qdot, q_p, qdot_p, qddot_p)
    assert isinstance(qddot, np.ndarray)
    assert qddot[0] == pytest.approx(-0.0)
    assert qddot[1] == pytest.approx(2.0)
    # towards obstacle from [0, 0] with [2.0, 0.0] -> accelerate in negative x
    q = np.array([0.0, 0.0])
    qdot = np.array([2.0, -0.0])
    qddot = planner.computeAction(q, qdot, q_p, qdot_p, qddot_p)
    assert isinstance(qddot, np.ndarray)
    assert qddot[0] == pytest.approx(-2.0)
    assert qddot[1] == pytest.approx(0.0)
    # away from obstacle from [0, 0] with [-2.0, 0.0] -> no action
    q = np.array([0.0, 0.0])
    qdot = np.array([-2.0, -0.0])
    qddot = planner.computeAction(q, qdot, q_p, qdot_p, qddot_p)
    assert isinstance(qddot, np.ndarray)
    assert qddot[0] == pytest.approx(0.0)
    assert qddot[1] == pytest.approx(0.0)
