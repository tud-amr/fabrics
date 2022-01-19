import pytest
import casadi as ca
import numpy as np
from fabrics.diffGeometry.geometry import Geometry
from fabrics.diffGeometry.energized_geometry import WeightedGeometry
from fabrics.diffGeometry.energy import FinslerStructure


@pytest.fixture
def simple_case():
    x = ca.SX.sym("x", 2)
    xdot = ca.SX.sym("xdot", 2)
    h = 0.5 / (x ** 2) * ca.norm_2(xdot) ** 2
    geo = Geometry(h=h, x=x, xdot=xdot)
    l_ex = 1.0 * ca.norm_2(xdot)
    lex = FinslerStructure(l_ex, x=x, xdot=xdot)
    l_e = 0.5 * ca.norm_2(x) * ca.norm_2(xdot)
    le = FinslerStructure(l_e, x=x, xdot=xdot)
    return geo, lex, le


def test_simple_case(simple_case):
    geo, lex, le = simple_case
    x = np.array([1.0, 0.2])
    xdot = np.array([1.0, -0.7])
    spec_ex = WeightedGeometry(g=geo, le=lex)
    spec_e = WeightedGeometry(g=geo, le=le)
    geo_e = Geometry(s=spec_e)
    spec_e_ex = WeightedGeometry(g=geo_e, le=lex)
    spec_ex.concretize()
    spec_e.concretize()
    spec_e_ex.concretize()
    M_e, f_e, xddot_e, alpha_e = spec_e.evaluate(x, xdot)
    M_ex, f_ex, xddot_ex, alpha_ex = spec_ex.evaluate(x, xdot)
    M_e_ex, f_e_ex, xddot_e_ex, alpha_e_ex = spec_e_ex.evaluate(x, xdot)
    assert xddot_e_ex[0] == pytest.approx(xddot_ex[0])
