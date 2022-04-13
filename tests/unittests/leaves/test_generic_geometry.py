import casadi as ca
import pytest

from fabrics.helpers.variables import Variables
from fabrics.leaves.generics.geometry import GenericGeometry


def test_attractor_generation():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    x_geometry = ca.SX.sym("x_geo", 2)
    var_q = Variables(
        state_variables={"q": q, "qdot": qdot}, parameters={"geometry_parameter": x_geometry}
    )
    GenericGeometry(var_q, q)


@pytest.fixture
def generic_geometry():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    x_geometry = ca.SX.sym("x_geo", 2)
    var_q = Variables(
        state_variables={"q": q, "qdot": qdot}, parameters={"geometry_parameter": x_geometry}
    )
    return GenericGeometry(var_q, q)


def test_set_geometry(generic_geometry: GenericGeometry):
    geometry_expression = (
        "ca.norm_2(x_geometry) + 0.1 * ca.log(1 + ca.exp(-2 * 10 * ca.norm_2(x_geometry)))"
    )
    generic_geometry.set_geometry(geometry_expression)


def test_set_metric(generic_geometry: GenericGeometry):
    finsler_expression = "ca.norm_2(xdot_geometry)**2 * 1/ca.norm_2(x_geometry)"
    generic_geometry.set_finsler_structure(finsler_expression)

