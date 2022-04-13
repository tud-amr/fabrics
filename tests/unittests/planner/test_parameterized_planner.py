import pytest
import numpy as np
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

def test_creation():
    ParameterizedFabricPlanner(2)

@pytest.fixture
def planner():
    return ParameterizedFabricPlanner(2)

def test_set_components(planner: ParameterizedFabricPlanner):
    fks = [planner.variables.position_variable()]
    fk = planner.variables.position_variable()
    planner.set_components(fks, fk)
    planner.concretize()

def test_compute_action(planner: ParameterizedFabricPlanner):
    fks = [planner.variables.position_variable()]
    fk = planner.variables.position_variable()
    planner.set_components(fks, fk)
    planner.concretize()
    q = np.zeros(2)
    qdot = np.zeros(2)
    x_goal = np.array([1.0, -1.0])
    x_obst = np.array([1.0, 0.2])
    qddot = planner.compute_action(q=q, qdot=qdot, x_goal=x_goal, x_obst_0=x_obst)
    assert isinstance(qddot, np.ndarray)
    assert qddot.size == 2
    assert qddot.shape == (2,)
