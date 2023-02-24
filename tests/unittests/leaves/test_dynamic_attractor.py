import casadi as ca
import pytest

from fabrics.helpers.variables import Variables
from fabrics.components.leaves.dynamic_attractor import GenericDynamicAttractor


def test_attractor_generation():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    x_goal = ca.SX.sym("x_goal", 2)
    var_q = Variables(
        state_variables={"q": q, "qdot": qdot}
    )
    GenericDynamicAttractor(var_q, q, "goal")


@pytest.fixture
def generic_dynamic_attractor():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    var_q = Variables(
        state_variables={"q": q, "qdot": qdot}
    )
    return GenericDynamicAttractor(var_q, q, "goal")


def test_set_potential(generic_dynamic_attractor: GenericDynamicAttractor):
    potential_expression = (
        "ca.norm_2(x) + 0.1 * ca.log(1 + ca.exp(-2 * 10 * ca.norm_2(x)))"
    )
    generic_dynamic_attractor.set_potential(potential_expression)


def test_set_metric(generic_dynamic_attractor: GenericDynamicAttractor):
    metric_expression = "((2.0 - 0.3) * ca.exp(-1 * (0.75 * ca.norm_2(x))**2) + 0.3) * ca.SX(np.identity(x.size()[0]))"
    generic_dynamic_attractor.set_metric(metric_expression)
