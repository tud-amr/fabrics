import pytest
import numpy as np
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner
from mpscenes.goals.goal_composition import GoalComposition

def test_creation():
    ParameterizedFabricPlanner(2, "pointRobot")

@pytest.fixture
def planner():
    return ParameterizedFabricPlanner(2, "pointRobot")

@pytest.fixture
def goal():
    goal_dict = {
        "subgoal0": {
            "weight": 1.0,
            "is_primary_goal": True,
            "indices": [0, 1],
            "parent_link": 0,
            "child_link": 2,
            "desired_position": [-4.0, 1.0],
            "epsilon": 0.15,
            "type": "staticSubGoal",
        }
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    return goal

def test_set_components(planner: ParameterizedFabricPlanner, goal: GoalComposition):
    planner.set_components(collision_links=[1], goal=goal)
    planner.concretize()

def test_compute_action(planner: ParameterizedFabricPlanner, goal: GoalComposition):
    planner.set_components(collision_links=[1], goal=goal)
    planner.concretize()
    q = np.zeros(2)
    qdot = np.zeros(2)
    x_goal = np.array([1.0, -1.0])
    x_obst = np.array([1.0, 0.2])
    qddot = planner.compute_action(
        q=q, qdot=qdot, x_goal_0=x_goal, weight_goal_0=np.array([1.0]),
        x_obst_0=x_obst, radius_body_1=np.array([0.5]), radius_obst_0=np.array([0.5])
    )
    assert isinstance(qddot, np.ndarray)
    assert qddot.size == 2
    assert qddot.shape == (2,)
    assert qddot[0] == pytest.approx(1.116237)
