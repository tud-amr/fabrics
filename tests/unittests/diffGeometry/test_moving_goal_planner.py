import pytest
import casadi as ca
import numpy as np

from fabrics.diffGeometry.diffMap import DifferentialMap, DynamicDifferentialMap
from fabrics.diffGeometry.geometry import Geometry

from forwardkinematics.planarFks.planarArmFk import PlanarArmFk

from fabrics.helpers.variables import Variables


@pytest.fixture
def movingGoalGeometry():
    x = ca.SX.sym("x", 2)
    xdot = ca.SX.sym("xdot", 2)
    var_x = Variables(state_variables={'x': x, 'xdot': xdot})
    x_rel = ca.SX.sym("x_rel", 2)
    xdot_rel = ca.SX.sym("xdot_rel", 2)
    var_x_rel = Variables(state_variables={'x_rel': x_rel, 'xdot_rel': xdot_rel})
    x_d = ca.SX.sym("x_d", 2)
    xdot_d = ca.SX.sym("xdot_d", 2)
    xddot_d = ca.SX.sym("xddot_d", 2)
    var_x_ref = Variables(parameters={'x_ref': x_d, 'xdot_ref': xdot_d, 'xddot_ref': xddot_d})

    # define geometry in relative coordinates x_rel
    # where psi is defined to have its minimum at x = x_d
    k_psi = 1
    psi = ca.norm_2(x-x_d)**2 * k_psi
    h_rel = ca.gradient(psi, x)
    geo_rel = Geometry(h=h_rel, var=var_x_rel)

    # define the relative transformation
    variables_dynamic = Variables(state_variables={'x': x, 'xdot': xdot}, parameters={'x_ref': x_d, 'xdot_ref': xdot_d, 'xddot_ref': xddot_d})
    dm_rel = DynamicDifferentialMap(variables_dynamic)
    geo = geo_rel.dynamic_pull(dm_rel)
    # Define second transform to configuration space
    n = 3
    q = ca.SX.sym("q", n)
    qdot = ca.SX.sym("qdot", n)
    var_q = Variables(state_variables={'q': q, 'qdot': qdot})
    planarArmFk = PlanarArmFk(n)
    phi_fk = planarArmFk.fk(q, n, positionOnly=True)
    dm_fk = DifferentialMap(phi_fk, var_q)
    geo_fk = geo.pull(dm_fk)
    return geo_rel, geo, geo_fk


def test_movingGoalGeometry(movingGoalGeometry):
    geo_rel, geo, geo_fk = movingGoalGeometry
    # example computes
    geo.concretize()
    geo_fk.concretize()
    x_0 = np.array([1.0, -1.0])
    xdot_0 = np.array([-0.0, 0.0])
    x_d_0 = np.array([-2.0, -1.0])
    xdot_d_0 = np.array([-0.0, 0.0])
    xddot_d_0 = np.array([2.0, 2.0])
    q_0 = np.array([-1.0, 0.0, 0.5])
    qdot_0 = np.array([1.0, 0.3, 0.5])

    # ATTENTION: geo_rel cannot be evaluated as the forcing term for the
    # is defined with respect to x and not x_rel
    # x_rel_0 = x_0 - x_d_0
    # xdot_rel_0 = xdot_0 - xdot_d_0
    # h, xddot_rel = geo_rel.evaluate(x_rel_0, xdot_rel_0)
    h, xddot = geo.evaluate(x=x_0, xdot=xdot_0, x_ref=x_d_0, xdot_ref=xdot_d_0, xddot_ref=xddot_d_0)
    h_fk, qddot = geo_fk.evaluate(q=q_0, qdot=qdot_0, x_ref=x_d_0, xdot_ref=xdot_d_0, xddot_ref=xddot_d_0)
    assert h[0] == pytest.approx(4.0)
    assert h[1] == pytest.approx(-2.0)

