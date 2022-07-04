import pytest
import casadi as ca
import numpy as np
from fabrics.planner.fabricPlanner import FabricPlanner
from fabrics.diffGeometry.diffMap import (
    DifferentialMap,
)
from fabrics.diffGeometry.energy import FinslerStructure, Lagrangian
from fabrics.diffGeometry.geometry import Geometry
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

from fabrics.helpers.variables import Variables

from MotionPlanningGoal.goalComposition import GoalComposition

@pytest.fixture
def goal_1d():
    goal_dict = {
        "subgoal0": {
            "m": 1,
            "w": 1.0,
            "prime": True,
            "indices": [0],
            "parent_link": 0,
            "child_link": 2,
            "desired_position": [-4.0],
            "epsilon": 0.15,
            "type": "staticSubGoal",
        }
    }
    goal = GoalComposition(name="goal", contentDict=goal_dict)
    return goal

@pytest.fixture
def goal_2d():
    goal_dict = {
        "subgoal0": {
            "m": 2,
            "w": 1.0,
            "prime": True,
            "indices": [0, 1],
            "parent_link": 0,
            "child_link": 2,
            "desired_position": [-4.0, 1.0],
            "epsilon": 0.15,
            "type": "staticSubGoal",
        }
    }
    goal = GoalComposition(name="goal", contentDict=goal_dict)
    return goal

@pytest.fixture
def parameterized_planner_1d(goal_1d):
    geometry_expression = "0.5 / (x ** 2) * xdot"
    collision_finsler = "0.5 / (x ** 2) * xdot**2"

    planner = ParameterizedFabricPlanner(
        1,
        'pointRobot',
        collision_geometry=geometry_expression,
        collision_finsler=collision_finsler,
        base_energy="0.5 * ca.dot(xdot, xdot)"
    )
    planner.set_components([planner._variables.position_variable()], {})
    planner.concretize()
    return planner

@pytest.fixture
def parameterized_planner_2d(goal_2d):
    geometry_expression = "0.5 / (x ** 2) * xdot"
    collision_finsler = "0.5 / (x ** 2) * xdot**2"

    planner_2d = ParameterizedFabricPlanner(
        2,
        'pointRobot',
        collision_geometry=geometry_expression,
        collision_finsler=collision_finsler,
        base_energy="0.5 * ca.dot(xdot, xdot)"
    )
    planner_2d.set_components([2], {})
    planner_2d.concretize()
    return planner_2d

@pytest.fixture
def planner_non_parameterized_1d():
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
    q_0 = 1
    phi = ca.fabs(q - q_0) - 1
    dm = DifferentialMap(phi, var_q)
    lg = 1 / x * xdot
    l = FinslerStructure(lg, var=var_x)
    h = 0.5 / (x ** 2) * xdot
    geo = Geometry(h=h, var=var_x)
    planner.addGeometry(dm, l, geo)
    planner.concretize()
    return planner


@pytest.fixture
def planner_non_parameterized_2d():
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
    phi = ca.norm_2(q - q0) - 1
    dm = DifferentialMap(phi, var_q)
    lg = 1 / x * xdot
    l = FinslerStructure(lg, var=var_x)
    h = 0.5 / (x ** 2) * xdot
    geo = Geometry(h=h, var=var_x)
    planner.addGeometry(dm, l, geo)
    planner.concretize()
    return planner


def test_simple_task(planner_non_parameterized_1d, parameterized_planner_1d):
    planner = planner_non_parameterized_1d
    planner_parameterized = parameterized_planner_1d
    print(planner_parameterized._variables)
    q_0 = np.array([1.0])
    for _ in range(10):
        q = -1 + 2 * np.random.random(1)
        qdot = -1 + 2 * np.random.random(1)
        qddot = planner.computeAction(q=q, qdot=qdot)
        qddot_parameterized = planner_parameterized.compute_action(
            q=q, qdot=qdot, x_obst_0=q_0,
            radius_obst_0=np.array([0.5]),
            radius_body_q=np.array([0.5])
        )
        assert qddot[0] == pytest.approx(qddot_parameterized[0])

def test_simple_task_2d(planner_non_parameterized_2d, parameterized_planner_2d):
    planner = planner_non_parameterized_2d
    planner_parameterized = parameterized_planner_2d
    q_0 = np.array([1.0, 0.0])
    for _ in range(10):
        q = -1 + 2 * np.random.random(2)
        qdot = -1 + 2 * np.random.random(2)
        qddot = planner.computeAction(q=q, qdot=qdot)
        qddot_parameterized = planner_parameterized.compute_action(
            q=q, qdot=qdot, x_obst_0=q_0,
            radius_obst_0=np.array([0.5]), radius_body_2=np.array([0.5])
        )
        assert qddot[0] == pytest.approx(qddot_parameterized[0])
        assert qddot[1] == pytest.approx(qddot_parameterized[1])
