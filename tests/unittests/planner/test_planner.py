import pytest
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

def test_creation():
    planner = ParameterizedFabricPlanner(2)

@pytest.fixture
def planner():
    return ParameterizedFabricPlanner(2)

def test_set_components(planner: ParameterizedFabricPlanner):
    fks = [planner.variables.position_variable()]
    fk = planner.variables.position_variable()
    planner.set_components(fks, fk)
