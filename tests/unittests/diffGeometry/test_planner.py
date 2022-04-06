import pytest
import casadi as ca
import numpy as np
from fabrics.planner.fabricPlanner import FabricPlanner
from fabrics.diffGeometry.diffMap import DifferentialMap, RelativeDifferentialMap
from fabrics.diffGeometry.energy import FinslerStructure, Lagrangian
from fabrics.diffGeometry.geometry import Geometry
from fabrics.diffGeometry.analyticSymbolicTrajectory import AnalyticSymbolicTrajectory

from fabrics.helpers.variables import Variables


@pytest.fixture
def simple_planner():
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    var = Variables(state_variables={"x": x, "xdot": xdot})
    l = 0.5 * xdot**2
    l_base = Lagrangian(l, var=var)
    geo_base = Geometry(h=ca.SX(0), var=var)
    planner = FabricPlanner(geo_base, l_base)
    return planner


@pytest.fixture
def simple_task():
    q = ca.SX.sym("q", 1)
    qdot = ca.SX.sym("qdot", 1)
    var_q = Variables(state_variables={"q": q, "qdot": qdot})
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    var_x = Variables(state_variables={"x": x, "xdot": xdot})
    l = 0.5 * ca.dot(qdot, qdot)
    l_base = Lagrangian(l, var=var_q)
    geo_base = Geometry(h=ca.SX(0), var=var_q)
    planner = FabricPlanner(geo_base, l_base)
    phi = ca.fabs(q - 1)
    dm = DifferentialMap(phi, var=var_q)
    s = -0.5 * (ca.sign(xdot) - 1)
    lg = 1 / x * s * xdot
    l = FinslerStructure(lg, var=var_x)
    h = 0.5 / (x ** 2) * xdot
    geo = Geometry(h=h, var=var_x)
    return planner, dm, l, geo


@pytest.fixture
def simple_2dtask():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    var_q = Variables(state_variables={"q": q, "qdot": qdot})
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    var_x = Variables(state_variables={"x": x, "xdot": xdot})
    l = 0.5 * ca.dot(qdot, qdot)
    l_base = Lagrangian(l, var=var_q)
    geo_base = Geometry(h=ca.SX(np.zeros(2)), var=var_q)
    planner = FabricPlanner(geo_base, l_base)
    q0 = np.array([1.0, 0.0])
    phi = ca.norm_2(q - q0)
    dm = DifferentialMap(phi, var=var_q)
    s = -0.5 * (ca.sign(xdot) - 1)
    lg = 1 / x * s * xdot
    l = FinslerStructure(lg, var=var_x)
    h = 0.5 / (x ** 2) * xdot
    geo = Geometry(h=h, var=var_x)
    return planner, dm, l, geo


@pytest.fixture
def variable_2dtask():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    var_q = Variables(state_variables={"q": q, "qdot": qdot})
    q_rel = ca.SX.sym("q_rel", 2)
    qdot_rel = ca.SX.sym("qdot_rel", 2)
    var_q_rel = Variables(state_variables={"q_rel": q_rel, "qdot_rel": qdot_rel})
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    var_x = Variables(state_variables={"x": x, "xdot": xdot})
    q_ref = ca.SX.sym("q_ref", 2)
    qdot_ref = ca.SX.sym("qdot_ref", 2)
    qddot_ref = ca.SX.sym("qddot_ref", 2)
    var_q_ref = Variables(parameters={"q_ref": q_ref, "qdot_ref": qdot_ref, "qddot_ref": qddot_ref})
    l = 0.5 * ca.dot(qdot, qdot)
    l_base = Lagrangian(l, var=var_q)
    geo_base = Geometry(h=ca.SX(np.zeros(2)), var=var_q)
    planner = FabricPlanner(geo_base, l_base)
    refTraj = AnalyticSymbolicTrajectory(ca.SX(np.identity(2)), 2, var=var_q_ref)
    dm_rel = RelativeDifferentialMap(var = var_q, refTraj=refTraj)
    phi = ca.norm_2(q_rel)
    dm = DifferentialMap(phi, var=var_q_rel)
    s = -0.5 * (ca.sign(xdot) - 1)
    lg = 1 / x * s * xdot
    l = FinslerStructure(lg, var=var_x)
    h = 0.5 / (x ** 2) * xdot
    geo = Geometry(h=h, var = var_x)
    return planner, dm, dm_rel, l, geo


def test_simple_planner(simple_planner):
    simple_planner.concretize()
    x = np.array([1.0])
    xdot = np.array([1.0])
    xddot = simple_planner.computeAction(x=x, xdot=xdot)
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
    qddot = planner.computeAction(q=q, qdot=qdot)
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
    qddot = planner.computeAction(q=q, qdot=qdot)
    assert isinstance(qddot, np.ndarray)
    assert qddot[0] == pytest.approx(-0.0)
    assert qddot[1] == pytest.approx(2.0)
    # towards obstacle from [0, 0] with [2.0, 0.0] -> accelerate in negative x
    q = np.array([0.0, 0.0])
    qdot = np.array([2.0, -0.0])
    qddot = planner.computeAction(q=q, qdot=qdot)
    assert isinstance(qddot, np.ndarray)
    assert qddot[0] == pytest.approx(-2.0)
    assert qddot[1] == pytest.approx(0.0)
    # away from obstacle from [0, 0] with [-2.0, 0.0] -> no action
    q = np.array([0.0, 0.0])
    qdot = np.array([-2.0, -0.0])
    qddot = planner.computeAction(q=q, qdot=qdot)
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
    qddot = planner.computeAction(q=q, qdot=qdot, q_ref=q_p, qdot_ref=qdot_p, qddot_ref=qddot_p)
    assert isinstance(qddot, np.ndarray)
    assert qddot[0] == pytest.approx(-0.0)
    assert qddot[1] == pytest.approx(2.0)
    # towards obstacle from [0, 0] with [2.0, 0.0] -> accelerate in negative x
    q = np.array([0.0, 0.0])
    qdot = np.array([2.0, -0.0])
    qddot = planner.computeAction(q=q, qdot=qdot, q_ref=q_p, qdot_ref=qdot_p, qddot_ref=qddot_p)
    assert isinstance(qddot, np.ndarray)
    assert qddot[0] == pytest.approx(-2.0)
    assert qddot[1] == pytest.approx(0.0)
    # away from obstacle from [0, 0] with [-2.0, 0.0] -> no action
    q = np.array([0.0, 0.0])
    qdot = np.array([-2.0, -0.0])
    qddot = planner.computeAction(q=q, qdot=qdot, q_ref=q_p, qdot_ref=qdot_p, qddot_ref=qddot_p)
    assert isinstance(qddot, np.ndarray)
    assert qddot[0] == pytest.approx(0.0)
    assert qddot[1] == pytest.approx(0.0)
