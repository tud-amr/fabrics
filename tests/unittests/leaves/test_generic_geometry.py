import casadi as ca
import pytest

from fabrics.helpers.variables import Variables
from fabrics.components.leaves.geometry import GenericGeometryLeaf


def test_geometry_generation():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    var_q = Variables(
        state_variables={"q": q, "qdot": qdot}
    )
    GenericGeometryLeaf(var_q, "simple_leaf", q)


@pytest.fixture
def generic_geometry():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    var_q = Variables(
        state_variables={"q": q, "qdot": qdot}
    )
    return GenericGeometryLeaf(var_q, "simple_leaf", q)


def test_set_geometry(generic_geometry: GenericGeometryLeaf):
    geometry_expression = (
        "ca.norm_2(x) + 0.1 * ca.log(1 + ca.exp(-2 * 10 * ca.norm_2(x)))"
    )
    generic_geometry.set_geometry(geometry_expression)


def test_set_metric(generic_geometry: GenericGeometryLeaf):
    finsler_expression = "ca.norm_2(xdot)**2 * 1/ca.norm_2(x)"
    generic_geometry.set_finsler_structure(finsler_expression)

