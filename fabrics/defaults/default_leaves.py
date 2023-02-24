import casadi as ca
import numpy as np

from fabrics.defaults.default_geometries import GoalGeometry
from fabrics.defaults.default_energies import GoalLagrangian
from fabrics.defaults.default_maps import GoalMap

from fabrics.diffGeometry.diffMap import DynamicDifferentialMap, DifferentialMap
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



