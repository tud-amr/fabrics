import pytest
import casadi as ca
import numpy as np
from fabrics.diffGeometry.energy import Lagrangian
from fabrics.diffGeometry.analyticSymbolicTrajectory import AnalyticSymbolicTrajectory
from fabrics.diffGeometry.diffMap import DynamicDifferentialMap

from fabrics.helpers.variables import Variables


@pytest.fixture
def rel_lag():
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    x_rel = ca.SX.sym("x_rel", 1)
    xdot_rel = ca.SX.sym("xdot_rel", 1)
    x_ref = ca.SX.sym("x_ref", 1)
    xdot_ref = ca.SX.sym("xdot_ref", 1)
    xddot_ref = ca.SX.sym("xddot_ref", 1)
    variables = Variables(state_variables={"x_rel": x_rel, "xdot_rel": xdot_rel})
    l_en = 0.5 * (xdot_rel) ** 2
    variables_dynamic = Variables(
        state_variables={"x": x, "xdot": xdot},
        parameters={"x_ref": x_ref, "xdot_ref": xdot_ref, "xddot_ref": xddot_ref},
    )
    dm = DynamicDifferentialMap(variables_dynamic)
    dynamic_lagrangian = Lagrangian(l_en, var=variables)
    dynamic_lagrangian.concretize()
    pulled_lagrangian = dynamic_lagrangian.dynamic_pull(dm)
    pulled_lagrangian.concretize()
    return dynamic_lagrangian, pulled_lagrangian


def test_rel_lag(rel_lag):
    l_rel, l_static = rel_lag
    x_ref = np.array([0.25])
    xdot_ref = np.array([0.5])
    xddot_ref = np.array([0.2])
    x = np.array([0.2])
    xdot = np.array([0.3])
    x_rel = x - x_ref
    xdot_rel = xdot - xdot_ref
    M_static, f_static, H_static = l_static.evaluate(
        x=x, xdot=xdot, x_ref=x_ref, xdot_ref=xdot_ref, xddot_ref=xddot_ref
    )
    M_rel, f_rel, H_rel = l_rel.evaluate(x_rel=x_rel, xdot_rel=xdot_rel)
    f_rel_pulled = f_rel - np.dot(M_rel, xddot_ref)
    assert M_static[0, 0] == pytest.approx(M_rel[0, 0], rel=1e-4)
    assert H_static == pytest.approx(H_rel, rel=1e-4)
    assert f_static[0] == pytest.approx(f_rel_pulled[0], rel=1e-4)
