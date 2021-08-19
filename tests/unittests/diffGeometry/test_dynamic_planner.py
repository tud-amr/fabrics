import pytest
import casadi as ca
import numpy as np

from optFabrics.planner.fabricPlanner import FabricPlanner
from optFabrics.diffGeometry.diffMap import DifferentialMap, RelativeDifferentialMap
from optFabrics.diffGeometry.energy import FinslerStructure, Lagrangian
from optFabrics.diffGeometry.geometry import Geometry

from casadiFk import casadiFk


@pytest.fixture
def movingGoalGeometry():
    x = ca.SX.sym("x", 2)
    xdot = ca.SX.sym("xdot", 2)
    x_rel = ca.SX.sym("x_rel", 2)
    xdot_rel = ca.SX.sym("xdot_rel", 2)
    x_d = ca.SX.sym("x_d", 2)
    xdot_d = ca.SX.sym("xdot_d", 2)
    xddot_d = ca.SX.sym("xddot_d", 2)


    # define geometry in relative coordinates x_rel
    # where psi is defined to have its minimum at x = x_d
    k_psi = 1
    psi = ca.norm_2(x-x_d)**2 * k_psi
    h_rel = ca.gradient(psi, x)
    geo_rel = Geometry(h=h_rel, x=x_rel, xdot=xdot_rel)

    # define the relative transformation
    dm_rel = RelativeDifferentialMap(q=x, qdot=xdot, q_p=x_d, qdot_p=xdot_d, qddot_p=xddot_d)
    geo = geo_rel.pull(dm_rel)
    # Define second transform to configuration space
    fks = []
    n = 3
    q = ca.SX.sym("q", n)
    qdot = ca.SX.sym("qdot", n)
    for i in range(1, n + 1):
        fks.append(ca.SX(casadiFk(q, i)[0:2]))
    phi_fk = fks[-1]
    dm_fk = DifferentialMap(phi_fk, q=q, qdot=qdot)
    geo_fk = geo.pull(dm_fk)
    return geo_rel, geo, geo_fk


def test_movingGoalGeometry(movingGoalGeometry):
    geo_rel, geo, geo_fk = movingGoalGeometry
    # example computes
    geo.concretize()
    geo_rel.concretize()
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
    h, xddot = geo.evaluate(x_0, xdot_0, x_d_0, xdot_d_0, xddot_d_0)
    h_fk, qddot = geo_fk.evaluate(q_0, qdot_0, x_d_0, xdot_d_0, xddot_d_0)
    assert h[0] == pytest.approx(4.0)
    assert h[1] == pytest.approx(-2.0)

