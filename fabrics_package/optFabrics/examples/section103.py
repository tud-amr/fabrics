import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import casadi as ca

from optFabrics.damper import Damper
from optFabrics.leaf import Leaf
from optFabrics.rootGeometry import RootGeometry
from optFabrics.functions import createMapping, generateLagrangian, generateEnergizer
from optFabrics.plottingGeometries import plotTraj, animate, plotMultipleTraj, plot, plotMulti

q = ca.SX.sym("q", 2)
qdot = ca.SX.sym("qdot", 2)


def setBoundaryLeaves():
    # Define boundary leaves
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym('xdot', 1)
    lam = 0.25
    a = np.array([0.4, 0.2, 20.0, 5.0])
    psi_lim = a[0] / (x**2) + a[1] * ca.log(ca.exp(-a[2] * (x - a[3])) + 1)
    s_lim = 0.5 * (ca.tanh(-10 * xdot) + 1)
    h_lim = xdot ** 2 * s_lim * lam * ca.gradient(psi_lim, x)
    L_lim = 0.5 * xdot**2 * lam/x
    (M_lim, f_lim) = generateLagrangian(L_lim, x, xdot, 'lim')
    # Define leaf
    q_min = -4.0
    phi_lim = q[0] - q_min
    lxlow = Leaf("xlim_low", phi_lim, M_lim, h_lim, x, xdot, q, qdot)
    # ---
    # Define leaf
    q_min = -4.0
    phi_lim = q[1] - q_min
    lylow = Leaf("ylim_low", phi_lim, M_lim, h_lim, x, xdot, q, qdot)
    # ---
    # Define leaf
    q_max = 4.0
    phi_lim = q_max - q[1]
    lyup = Leaf("xlim_up", phi_lim, M_lim, h_lim, x, xdot, q, qdot)
    # ---
    # Define leaf
    q_max = 4.0
    phi_lim = q_max - q[0]
    lxup = Leaf("ylim_up", phi_lim, M_lim, h_lim, x, xdot, q, qdot)
    # ---
    return (lxup, lxlow, lyup, lylow)

def createCollisionAvoidance():
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym('xdot', 1)
    lam = 0.25
    a = np.array([0.4, 0.2, 20.0, 5.0])
    psi_col = a[0] / (x**2) + a[1] * ca.log(ca.exp(-a[2] * (x - a[3])) + 1)
    s_col = 0.5 * (ca.tanh(-10 * xdot) + 1)
    h_col = xdot ** 2 * lam * ca.gradient(psi_col, x)
    L_col = 0.5 * s_col * xdot**2 * lam/x
    (M_col, f_col) = generateLagrangian(L_col, x, xdot, 'col')
    q_obst = np.array([0.0, 0.0])
    r_obst = 1.0
    phi_col = ca.norm_2(q - q_obst) / r_obst - 1
    lcol = Leaf("col_avo", phi_col, M_col, h_col, x, xdot, q, qdot)
    return lcol

def forcingLeaf():
    x = ca.SX.sym("x", 2)
    xdot = ca.SX.sym("xdot", 2)
    q_d = np.array([-2.5, -3.75])
    k = 5.0
    alpha_psi = 10.0
    alpha_m = 0.75
    m = np.array([0.3, 2.0])

    phi = q - q_d
    psi = k * (ca.norm_2(x) + 1/alpha_psi * ca.log(1 + ca.exp(-2*alpha_psi * ca.norm_2(x))))
    M_forcing = ((m[1] - m[0]) * ca.exp(-(alpha_m * ca.norm_2(x))**2) + m[0]) * np.identity(2)
    h_forcing = ca.mtimes(M_forcing, ca.gradient(psi, x))
    lforcing = Leaf("forcing", phi, M_forcing, h_forcing, x, xdot, q, qdot)
    return lforcing

def createDamper(forcingLeave):
    ale = ca.SX.sym("ale", 1)
    alex = ca.SX.sym("alex", 1)
    ele = ca.SX.sym('ele', 1)
    elex = ca.SX.sym('elex', 1)
    a_eta = 0.5
    a_beta = 0.5
    a_shift = 0.5
    r = 1.5
    b = np.array([0.01, 6.5])
    q_d = np.array([-2.5, -3.75])
    beta_switch = 0.5 * (ca.tanh(-a_beta * (ca.norm_2(q - q_d) - r)) + 1)
    beta = beta_switch * b[1] + b[0] + ca.fmax(0.0, alex - ale)
    eta = 0.5 * (ca.tanh(-a_eta*(ele - elex) - a_shift) + 1)
    le = 0.5 * ca.norm_2(qdot)**2
    lex = 0.25 * ca.norm_2(qdot)**2
    # Functions
    beta_fun = ca.Function("beta", [q, qdot, ale, alex], [beta])
    eta_fun = ca.Function("eta", [ele, elex], [eta])
    damper = Damper(forcingLeave, beta_fun, eta_fun, le, lex, q, qdot)
    return damper

def main():
    (lxup, lxlow, lyup, lylow) = setBoundaryLeaves()
    lforcing = forcingLeaf()
    lcol = createCollisionAvoidance()
    damper = createDamper(lforcing)
    rg = RootGeometry([], 2)
    rg_forced = RootGeometry([lforcing], 2, damper=damper)
    rg_limits = RootGeometry([lyup, lylow, lxup, lxlow], 2)
    rg_forced_limits = RootGeometry([lyup, lylow, lxup, lxlow, lforcing], 2, damper=damper)
    rg_col = RootGeometry([lcol], 2)
    rg_col_limits = RootGeometry([lcol, lyup, lylow, lxup, lxlow], 2)
    rg_col_limits_forced = RootGeometry([lcol, lyup, lylow, lxup, lxlow, lforcing], 2, damper=damper)
    geos = [rg, rg_forced, rg_limits, rg_forced_limits, rg_col_limits, rg_col_limits_forced]
    #geos = [rg_col, rg_col_limits]
    # solve
    dt = 0.05
    T = 16.0
    sols = []
    aniSols = []
    x0 = np.array([2.0, 3.0])
    x0dot_norm = 1.5
    init_angles = [((i+7.13) * np.pi)/7 for i in range(14)]
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
    fig, ax = plt.subplots(3, 2, figsize=(10, 15))
    plotMulti(sols, aniSols, fig, ax)

if __name__ == "__main__":
    main()
