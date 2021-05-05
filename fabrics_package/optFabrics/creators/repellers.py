import numpy as np
from optFabrics.leaf import *

def createCollisionAvoidance(q, qdot, fk, x_obst, r_obst, lam=0.25, a=np.array([0.4, 0.2, 20.0, 5.0])):
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym('xdot', 1)
    phi = ca.norm_2(fk - x_obst) / r_obst - 1
    dm = DiffMap("attractor", phi, q, qdot, x, xdot)
    psi_col = a[0] / (x**2) + a[1] * ca.log(ca.exp(-a[2] * (x - a[3])) + 1)
    s_col = 0.5 * (ca.tanh(-10 * xdot) + 1)
    h = xdot ** 2 * lam * ca.gradient(psi_col, x)
    le = 0.5 * s_col * xdot**2 * lam/x
    lcol = GeometryLeaf("col_avo", dm, le, h)
    return lcol

def createTimeVariantCollisionAvoidance(q, qdot, fk, t, x_obst, r_obst, lam=0.25, a=np.array([0.4, 0.2, 20.0, 5.0])):
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym('xdot', 1)
    phi = ca.norm_2(fk - x_obst) / r_obst - 1
    dm = TimeVariantDiffMap("attractor", phi, q, qdot, x, xdot, t)
    psi_col = a[0] / (x**2) + a[1] * ca.log(ca.exp(-a[2] * (x - a[3])) + 1)
    s_col = 0.5 * (ca.tanh(-10 * xdot) + 1)
    h = xdot ** 2 * lam * ca.gradient(psi_col, x)
    le = 0.5 * s_col * xdot**2 * lam/x
    lcol = GeometryLeaf("col_avo", dm, le, h)
    return lcol

def createJointLimits(q, qdot, upper_lim, lower_lim, a=np.array([0.4, 0.2, 20.0, 5.0]), lam=0.25):
    n = len(lower_lim)
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym('xdot', 1)
    psi_lim = a[0] / (x**2) + a[1] * ca.log(ca.exp(-a[2] * (x - a[3])) + 1)
    s_lim = 0.5 * (ca.tanh(-10 * xdot) + 1)
    h = xdot ** 2 * s_lim * lam * ca.gradient(psi_lim, x)
    le = 0.5 * xdot**2 * lam/x
    leaves = []
    for i in range(n):
        q_min = lower_lim[i]
        q_max = upper_lim[i]
        phi_min = q[i] - q_min
        phi_max = q_max - q[i]
        dm_min = DiffMap("limitmin_" + str(i), phi_min, q, qdot, x, xdot)
        dm_max = DiffMap("limitmax_" + str(i), phi_max, q, qdot, x, xdot)
        lqmin = GeometryLeaf("qmin_" + str(i), dm_min, le, h)
        lqmax = GeometryLeaf("qmax_" + str(i), dm_max, le, h)
        leaves.append(lqmin)
        leaves.append(lqmax)
    return leaves
