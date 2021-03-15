import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import casadi as ca

from optFabrics.damper import Damper
from optFabrics.leaf import Leaf
from optFabrics.rootGeometry import RootGeometry
from optFabrics.functions import createMapping, generateLagrangian, generateEnergizer
from optFabrics.plottingGeometries import plotTraj, animate, plotMultipleTraj, plot, plotMulti

from optFabrics.leaf import createAttractor, createCollisionAvoidance, createJointLimits


def forwardKinematics(q):
    x = q
    return x

def limits():
    up = np.array([4, 4])
    low = -up
    return (up, low)

def main():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    fk = forwardKinematics(q)
    lim_up, lim_low = limits()
    lcol = createCollisionAvoidance(q, qdot, fk, np.array([0.0, 0.0]), 1.0)
    limitLeaves = createJointLimits(q, qdot, lim_up, lim_low)
    x = ca.SX.sym("x", 2)
    xdot = ca.SX.sym("xdot", 2)
    x_d = np.array([-2.5, -2.5])
    lforcing = createAttractor(q, qdot, x, xdot, x_d, fk)
    rg = RootGeometry([], 2)
    rg_forced = RootGeometry([lforcing], 2)
    rg_limits = RootGeometry(limitLeaves, 2)
    rg_forced_limits = RootGeometry(limitLeaves +  [lforcing], 2)
    rg_col = RootGeometry([lcol], 2)
    rg_col_limits = RootGeometry(limitLeaves + [lcol], 2)
    rg_col_limits_forced = RootGeometry(limitLeaves + [lforcing, lcol], 2)
    geos = [rg, rg_forced, rg_limits, rg_forced_limits, rg_col_limits, rg_col_limits_forced]
    # solve
    dt = 0.01
    T = 16.0
    sols = []
    aniSols = []
    x0 = np.array([2.0, 3.0])
    x0dot_norm = 1.5
    init_angles = [5.0 * np.pi/4.0 + (i * np.pi)/7 for i in range(16)]
    for geo in geos:
        geoSols = []
        for i, a in enumerate(init_angles):
            print("Solving for ", a)
            x0dot = np.array([np.cos(a), np.sin(a)]) * x0dot_norm
            z0 = np.concatenate((x0, x0dot))
            sol = geo.computePath(z0, dt, T)
            geoSols.append(sol)
            if i == 0:
                aniSols.append(sol)
        sols.append(geoSols)
    fig, ax = plt.subplots(3, 2, figsize=(7, 13))
    plotMulti(sols, aniSols, fig, ax)

if __name__ == "__main__":
    main()
