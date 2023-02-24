import casadi as ca
import pytest

from fabrics.helpers.variables import Variables
from fabrics.components.leaves.dynamic_geometry import DynamicObstacleLeaf


def test_dynamic_obstacle_generation():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    var_q = Variables(
        state_variables={"q": q, "qdot": qdot}
    )
    collision_link = "link1"
    DynamicObstacleLeaf(var_q, q, "obstacle_1", collision_link)


@pytest.fixture
def generic_dynamic_obstacle():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    var_q = Variables(
        state_variables={"q": q, "qdot": qdot}
    )
    collision_link = "link1"
    return DynamicObstacleLeaf(var_q, q, "obstacle_1", collision_link)


def test_set_geometry(generic_dynamic_obstacle: DynamicObstacleLeaf):
    geometry_expression = (
        "ca.norm_2(x) + 0.1 * ca.log(1 + ca.exp(-2 * 10 * ca.norm_2(x)))"
    )
    generic_dynamic_obstacle.set_geometry(geometry_expression)


def test_set_metric(generic_dynamic_obstacle: DynamicObstacleLeaf):
    finsler_expression = "ca.norm_2(xdot)**2 * 1/ca.norm_2(x)"
    generic_dynamic_obstacle.set_finsler_structure(finsler_expression)
