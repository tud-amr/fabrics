import pytest
import numpy as np
import os
import yaml

from forwardkinematics.fksCommon.fk_creator import FkCreator

from mpscenes.goals.goal_composition import GoalComposition

from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

def test_creation():
    fk = FkCreator("pointRobot").fk()
    ParameterizedFabricPlanner(2, fk)

@pytest.fixture
def planner():
    fk = FkCreator("pointRobot").fk()
    return ParameterizedFabricPlanner(2, fk)

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

def test_load_configuration(planner: ParameterizedFabricPlanner):
    config_file = os.path.join(os.path.dirname(__file__), "planner_config.yaml")
    with open(config_file, 'r') as config_file:
        config = yaml.safe_load(config_file)
        config_problem = config['problem']
        config_fabrics = config['fabrics']
    planner.load_fabrics_configuration(config_fabrics)
    planner.load_problem_configuration(config_problem)

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
