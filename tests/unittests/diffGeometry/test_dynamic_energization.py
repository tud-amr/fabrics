import pytest
import casadi as ca
from typing import Tuple
import numpy as np
from fabrics.diffGeometry.geometry import Geometry
from fabrics.diffGeometry.energized_geometry import (
    WeightedGeometry,
)
from fabrics.diffGeometry.energy import FinslerStructure
from fabrics.diffGeometry.diffMap import DifferentialMap, DynamicDifferentialMap
from fabrics.helpers.variables import Variables

eps = 1e-5

def symvar(name: str, dim: int):
    return ca.SX.sym(name, dim)


@pytest.fixture
def energization_example() -> Tuple[Geometry, FinslerStructure]:
    x_rel = symvar("x_rel", 2)
    xdot_rel = symvar("xdot_rel", 2)
    h = 0.5 / (x_rel**2) * ca.norm_2(xdot_rel) ** 2
    var_rel = Variables(state_variables={"x_rel": x_rel, "xdot_rel": xdot_rel})
    geo = Geometry(h=h, var=var_rel)
    l = 1.3 * ca.norm_2(xdot_rel) * ca.norm_2(x_rel)
    le = FinslerStructure(l, var=var_rel)
    return geo, le


@pytest.fixture
def dynamic_map() -> DynamicDifferentialMap:
    x_ref = symvar("x_ref", 2)
    xdot_ref = symvar("xdot_ref", 2)
    xddot_ref = symvar("xddot_ref", 2)
    x = symvar("x", 2)
    xdot = symvar("xdot", 2)
    var = Variables(
        state_variables={"x": x, "xdot": xdot},
        parameters={"x_ref": x_ref, "xdot_ref": xdot_ref, "xddot_ref": xddot_ref},
    )
    return DynamicDifferentialMap(var)


def test_dynamic_energization(energization_example, dynamic_map):
    dm = dynamic_map
    geo = energization_example[0]
    lag = energization_example[1]
    wg = WeightedGeometry(g=geo, le=lag)
    wg_pulled = wg.dynamic_pull(dm)

    lag_pulled = lag.dynamic_pull(dm)
    geo_pulled = geo.dynamic_pull(dm)
    wg_pulled_1 = WeightedGeometry(g=geo_pulled, le=lag_pulled)

    wg.concretize()
    wg_pulled.concretize()
    wg_pulled_1.concretize()

    x_np = np.array([1.2, 0.1])
    xdot_np = np.array([0.1, 0.2])
    x_ref_np = np.array([1.4, 0.3])
    xdot_ref_np = np.array([-1.2, 2.1])
    xddot_ref_np = np.array([1.2, -1.0])

    x_rel_np = x_np - x_ref_np
    xdot_rel_np = xdot_np - xdot_ref_np

    M_rel, f_rel, xddot_rel, alpha_rel = wg.evaluate(
        x_rel=x_rel_np,
        xdot_rel=xdot_rel_np,
    )

    f_rel_pulled = f_rel - np.dot(M_rel, xddot_ref_np)
    xddot_rel_pulled = xddot_rel + xddot_ref_np

    M, f, xddot, alpha = wg_pulled.evaluate(
        x=x_np,
        xdot=xdot_np,
        x_ref=x_ref_np,
        xdot_ref=xdot_ref_np,
        xddot_ref=xddot_ref_np,
    )
    M_1, f_1, xddot_1, alpha_1 = wg_pulled_1.evaluate(
        x=x_np,
        xdot=xdot_np,
        x_ref=x_ref_np,
        xdot_ref=xdot_ref_np,
        xddot_ref=xddot_ref_np,
    )
    assert M[0, 0] == pytest.approx(M_1[0, 0])
    assert M[0, 1] == pytest.approx(M_1[0, 1])
    assert M[1, 0] == pytest.approx(M_1[1, 0])
    assert M[1, 1] == pytest.approx(M_1[1, 1])
    assert M[0, 0] == pytest.approx(M_rel[0, 0])
    assert M[0, 1] == pytest.approx(M_rel[0, 1])
    assert M[1, 0] == pytest.approx(M_rel[1, 0])
    assert M[1, 1] == pytest.approx(M_rel[1, 1])

    assert f[0] == pytest.approx(f_1[0])
    assert f[1] == pytest.approx(f_1[1])
    assert f[0] == pytest.approx(f_rel_pulled[0], rel=eps)
    assert f[1] == pytest.approx(f_rel_pulled[1], rel=eps)

    assert xddot[0] == pytest.approx(xddot_1[0], rel=eps)
    assert xddot[1] == pytest.approx(xddot_1[1], rel=eps)
    assert xddot[0] == pytest.approx(xddot_rel_pulled[0], rel=eps)
    assert xddot[1] == pytest.approx(xddot_rel_pulled[1], rel=eps)

    assert alpha == pytest.approx(alpha_1)
    assert alpha == pytest.approx(alpha_rel)
