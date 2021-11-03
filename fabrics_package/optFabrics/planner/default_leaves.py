import casadi as ca
import numpy as np

from optFabrics.planner.default_geometries import GoalGeometry
from optFabrics.planner.default_energies import GoalLagrangian
from optFabrics.planner.default_maps import GoalMap

from optFabrics.diffGeometry.diffMap import RelativeDifferentialMap, DifferentialMap
from optFabrics.diffGeometry.referenceTrajectory import AnalyticTrajectory


def defaultAttractor(q: ca.SX, qdot: ca.SX, goal: np.ndarray, fk: ca.SX):
    x = ca.SX.sym("x_psi", fk.size()[0])
    xdot = ca.SX.sym("xdot_psi", fk.size()[0])
    dm = GoalMap(q, qdot, fk, goal)
    lag = GoalLagrangian(x, xdot)
    geo = GoalGeometry(x, xdot)
    return dm, lag, geo, x, xdot


def defaultDynamicAttractor(q: ca.SX, qdot: ca.SX, fk: ca.SX, refTraj: AnalyticTrajectory, **kwargs):
    p = {"k_psi": 20}
    for key in p.keys():
        if key in kwargs:
            p[key] = kwargs.get(key)
    x = ca.SX.sym("x", 2)
    xdot = ca.SX.sym("xdot", 2)
    x_rel = ca.SX.sym("x_rel", 2)
    xdot_rel = ca.SX.sym("xdot_rel", 2)
    # relative systems
    dm_rel = RelativeDifferentialMap(q=x, qdot=xdot, refTraj=refTraj)
    lag_psi = GoalLagrangian(x_rel, xdot_rel).pull(dm_rel)
    geo_psi = GoalGeometry(x_rel, xdot_rel, k_psi=p['k_psi']).pull(dm_rel)
    phi_psi = fk
    dm_psi = DifferentialMap(phi_psi, q=q, qdot=qdot)
    return dm_psi, lag_psi, geo_psi, x, xdot

