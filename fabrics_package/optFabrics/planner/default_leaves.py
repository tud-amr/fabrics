import casadi as ca
import numpy as np

from optFabrics.planner.default_geometries import GoalGeometry
from optFabrics.planner.default_energies import GoalLagrangian
from optFabrics.planner.default_maps import GoalMap

from optFabrics.diffGeometry.diffMap import RelativeDifferentialMap, DifferentialMap


def defaultAttractor(q: ca.SX, qdot: ca.SX, goal: np.ndarray, fk: ca.SX):
    x = ca.SX.sym("x_psi", fk.size()[0])
    xdot = ca.SX.sym("xdot_psi", fk.size()[0])
    dm = GoalMap(q, qdot, fk, goal)
    lag = GoalLagrangian(x, xdot)
    geo = GoalGeometry(x, xdot)
    return dm, lag, geo, x, xdot


def defaultDynamicAttractor(q: ca.SX, qdot: ca.SX, fk: ca.SX):
    x = ca.SX.sym("x", 2)
    xdot = ca.SX.sym("xdot", 2)
    x_g = ca.SX.sym("x_g", 2)
    xdot_g = ca.SX.sym("xdot_g", 2)
    xddot_g = ca.SX.sym("xdot_g", 2)
    x_rel = ca.SX.sym("x_rel", 2)
    xdot_rel = ca.SX.sym("xdot_rel", 2)
    # relative systems
    lag_psi = GoalLagrangian(x, xdot)
    geo_rel = GoalGeometry(x_rel, xdot_rel, k_psi=20)
    dm_rel = RelativeDifferentialMap(q=x, qdot=xdot, q_p=x_g, qdot_p=xdot_g, qddot_p=xddot_g)
    geo_psi = geo_rel.pull(dm_rel)
    phi_psi = fk
    dm_psi = DifferentialMap(phi_psi, q=q, qdot=qdot)
    return dm_psi, lag_psi, geo_psi, x, xdot, xdot_g

