import casadi as ca
import pytest

from fabrics.helpers.variables import Variables
from fabrics.components.leaves.attractor import GenericAttractor


def test_attractor_generation():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    x_goal = ca.SX.sym("x_goal", 2)
    var_q = Variables(
        state_variables={"q": q, "qdot": qdot}, parameters={"x_goal": x_goal}
    )
    GenericAttractor(var_q, q, "goal")


@pytest.fixture
def generic_attractor():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    x_goal = ca.SX.sym("x_psi", 2)
    var_q = Variables(
        state_variables={"q": q, "qdot": qdot}, parameters={"x_goal": x_goal}
    )
    return GenericAttractor(var_q, q, "goal")


def test_set_potential(generic_attractor: GenericAttractor):
    potential_expression = (
        "ca.norm_2(x) + 0.1 * ca.log(1 + ca.exp(-2 * 10 * ca.norm_2(x)))"
    )
    generic_attractor.set_potential(potential_expression)


def test_set_metric(generic_attractor: GenericAttractor):
    metric_expression = "((2.0 - 0.3) * ca.exp(-1 * (0.75 * ca.norm_2(x))**2) + 0.3) * ca.SX(np.identity(x.size()[0]))"
    generic_attractor.set_metric(metric_expression)
