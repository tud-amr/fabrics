import casadi as ca
import numpy as np

from optFabrics.planner.default_geometries import GoalGeometry
from optFabrics.planner.default_energies import GoalLagrangian
from optFabrics.planner.default_maps import GoalMap


def defaultAttractor(q: ca.SX, qdot: ca.SX, goal: np.ndarray, fk: ca.SX):
    x = ca.SX.sym("x_psi", 2)
    xdot = ca.SX.sym("xdot_psi", 2)
    dm = GoalMap(q, qdot, fk, goal)
    lag = GoalLagrangian(x, xdot)
    geo = GoalGeometry(x, xdot)
    return dm, lag, geo, x, xdot

