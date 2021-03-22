import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import casadi as ca

from optFabrics.damper import Damper, RootDamper, createRootDamper
from optFabrics.leaf import Leaf
from optFabrics.rootGeometry import RootGeometry
from optFabrics.functions import createMapping, generateLagrangian, generateEnergizer
from optFabrics.plottingGeometries import plotTraj, animate, plotMultipleTraj, plot, plotMulti
from optFabrics.diffMap import DiffMap

from optFabrics.leaf import createAttractor, createCollisionAvoidance, createJointLimits, createTimeVariantAttractor

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
    x_obst = np.array([0.0, 0.2])
    r_obst = 2.0
    lcol = createCollisionAvoidance(q, qdot, fk, x_obst, r_obst)
    limitLeaves = createJointLimits(q, qdot, lim_up, lim_low)
    x = ca.SX.sym("x", 2)
    xdot = ca.SX.sym("xdot", 2)
    t = ca.SX.sym("t", 1)
    x_d = 2.0 * ca.vertcat(2.0 * ca.cos(t) , -0.2 * ca.sin(t))
    lforcing = createTimeVariantAttractor(q, qdot, x, xdot, x_d, t, fk, k=5.0)
    x_d = np.array([-2.0, 1.0])
    lforcingstatic = createAttractor(q, qdot, x, xdot, x_d, fk, k=5.0)
    # damper
    x_ex = ca.SX.sym("x_ex", 2)
    xdot_ex = ca.SX.sym("xdot_ex", 2)
    phi_ex = q
    diffMap_ex = DiffMap("ex_map", phi_ex, q, qdot, x_ex, xdot_ex)
    rootDamper = createRootDamper(q, qdot, x, diffMap_ex, x_ex, xdot_ex)
    le_root = 1.0/2.0 * ca.dot(qdot, qdot)
    rg = RootGeometry([], le_root, 2)
    rg_forced = RootGeometry([lforcing], le_root, 2, damper=rootDamper)
    rg_forced_static = RootGeometry([lforcingstatic], le_root, 2, damper=rootDamper)
    geos = [rg_forced, rg_forced_static]
    # solve
    dt = 0.01
    T = 20.0
    sols = []
    aniSols = []
    x0 = np.array([2.0, 3.0])
    x0dot_norm = 1.5
    init_angles = [1.0 * np.pi/5.0 + (i * np.pi)/5 for i in range(10)]
    for geo in geos:
        geoSols = []
        for i, a in enumerate(init_angles):
            print("Solving for ", a)
            x0dot = np.array([np.cos(a), np.sin(a)]) * x0dot_norm
            z0 = np.concatenate((x0, x0dot))
            sol = geo.computePath(z0, dt, T)
            geoSols.append(sol)
            if i == 6:
                aniSols.append(sol)
        sols.append(geoSols)
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    obst = plt.Circle(x_obst, radius=r_obst, color='r')
    obst2 = plt.Circle(x_obst, radius=r_obst, color='r')
    #ax[1][0].add_patch(obst)
    #ax[1][1].add_patch(obst2)
    plotMulti(sols, aniSols, fig, ax)

if __name__ == "__main__":
    main()
