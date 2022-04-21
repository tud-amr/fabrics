import casadi as ca
import numpy as np

from fabrics.defaults.default_geometries import GoalGeometry
from fabrics.defaults.default_energies import GoalLagrangian
from fabrics.defaults.default_maps import GoalMap

from fabrics.diffGeometry.diffMap import RelativeDifferentialMap, DifferentialMap
from fabrics.diffGeometry.analyticSymbolicTrajectory import AnalyticSymbolicTrajectory

from fabrics.helpers.variables import Variables

def defaultAttractor(var_q: Variables, goal: np.ndarray, fk: ca.SX, **kwargs):
    p = {"k_psi": 10}
    for key in p.keys():
        if key in kwargs:
            p[key] = kwargs.get(key)
    x = ca.SX.sym("x_psi", fk.size()[0])
    xdot = ca.SX.sym("xdot_psi", fk.size()[0])
    var_x = Variables(state_variables={'x': x, 'xdot': xdot})
    dm = GoalMap(var_q, fk, goal)
    lag = GoalLagrangian(var_x)
    geo = GoalGeometry(var_x, k_psi=p['k_psi'])
    return dm, lag, geo, var_x


def defaultDynamicAttractor(var_q: Variables, fk: ca.SX, refTraj: AnalyticSymbolicTrajectory, **kwargs):
    p = {"k_psi": 20}
    for key in p.keys():
        if key in kwargs:
            p[key] = kwargs.get(key)
    x = ca.SX.sym("x", refTraj.n())
    xdot = ca.SX.sym("xdot", refTraj.n())
    var_x = Variables(state_variables={'x': x, 'xdot': xdot})
    x_goal_rel = ca.SX.sym("x_goal_rel", refTraj.n())
    xdot_goal_rel = ca.SX.sym("xdot_goal_rel", refTraj.n())
    var_goal_rel = Variables(state_variables={'x_goal_rel': x_goal_rel, 'xdot_goal_rel': xdot_goal_rel})
    # relative systems
    dm_rel = RelativeDifferentialMap(var=var_x, refTraj=refTraj)
    lag_psi = GoalLagrangian(var_goal_rel).pull(dm_rel)
    geo_psi = GoalGeometry(var_goal_rel, k_psi=p['k_psi']).pull(dm_rel)
    phi_psi = fk
    dm_psi = DifferentialMap(phi_psi, var=var_q)
    return dm_psi, lag_psi, geo_psi, var_x

