import numpy as np
from optFabrics.leaf import *

def createAttractor(q, qdot, x, xdot, x_d, fk,  k=5.0, a_psi=10.0, a_m = 0.75, m=np.array([0.3, 2.0])):
    n = x.size(1)
    phi = fk - x_d
    dm = DiffMap("attractor", phi, q, qdot, x, xdot)
    psi = k * (ca.norm_2(x) + 1/a_psi * ca.log(1 + ca.exp(-2*a_psi * ca.norm_2(x))))
    M = ((m[1] - m[0]) * ca.exp(-(a_m * ca.norm_2(x))**2) + m[0]) * np.identity(n)
    le = ca.dot(xdot, ca.mtimes(M, xdot))
    damper = createDamper(x, xdot, le)
    lforcing = ForcingLeaf("attractor", dm, le, psi)
    return lforcing

def createGroundRobotAttractor(q, qdot, x, xdot, x_d, fk,  k=5.0, a_psi=10.0, a_m = 0.75, m=np.array([0.3, 2.0])):
    n = x.size(1)
    phi = fk - x_d
    h = ca.vertcat(ca.cos(q[2]), ca.sin(q[2]))
    d = ca.vertcat(x_d[0] - fk[0], x_d[1] - fk[1])
    angle = np.arctan(h[1]/h[0]) - np.arctan(d[1]/d[0])
    angleFiltered = (0.5 * (ca.tanh(10.0 * (ca.fabs(angle) - 0.7)) + 1)) * angle
    phi = ca.vertcat(phi, angleFiltered)
    dm = DiffMap("attractor", phi, q, qdot, x, xdot)
    psi = k * (ca.norm_2(x) + 1/a_psi * ca.log(1 + ca.exp(-2*a_psi * ca.norm_2(x))))
    M = ((m[1] - m[0]) * ca.exp(-(a_m * ca.norm_2(x))**2) + m[0]) * np.identity(n)
    le = ca.dot(xdot, ca.mtimes(M, xdot))
    damper = createDamper(x, xdot, le)
    lforcing = ForcingLeaf("attractor", dm, le, psi)
    return lforcing

def createRotationAttractor(q, qdot, x, xdot, fk1, fk2, diff=0.0, k=5.0, a_psi=10.0, a_m = 0.75, m=np.array([0.3, 2.0])):
    n = 1
    phi = fk2 - fk1 - diff
    dm = DiffMap("attractor", phi, q, qdot, x, xdot)
    psi = k * (ca.norm_2(x) + 1/a_psi * ca.log(1 + ca.exp(-2*a_psi * ca.norm_2(x))))
    M = ((m[1] - m[0]) * ca.exp(-(a_m * ca.norm_2(x))**2) + m[0]) * np.identity(n)
    le = ca.dot(xdot, ca.mtimes(M, xdot))
    damper = createDamper(x, xdot, le)
    lforcing = ForcingLeaf("attractor", dm, le, psi)
    return lforcing

def createTimeVariantAttractor(q, qdot, x, xdot, x_d, t, fk,  k=5.0, a_psi=10.0, a_m = 0.75, m=np.array([0.3, 2.0])):
    n = x.size(1)
    phi = fk - x_d
    dm = TimeVariantDiffMap("attractor", phi, q, qdot, x, xdot, t)
    psi = k * (ca.norm_2(x) + 1/a_psi * ca.log(1 + ca.exp(-2*a_psi * ca.norm_2(x))))
    M = ((m[1] - m[0]) * ca.exp(-(a_m * ca.norm_2(x))**2) + m[0]) * np.identity(n)
    le = ca.dot(xdot, ca.mtimes(M, xdot))
    damper = createDamper(x, xdot, le)
    lforcing = ForcingLeaf("attractor", dm, le, psi)
    return lforcing

def createDynamicAttractor(q, qdot, x, xdot, xd_ca, xd_t, t_ca, fk, k=5.0):
    n = x.size(1)
    phi = fk
    dm = DiffMap("attractor", phi, q, qdot, x, xdot)
    psi = k * ca.norm_2(x - xd_ca)**2
    M = np.identity(n)
    le = 0.5 * ca.dot(xdot, ca.mtimes(M, xdot))
    ldynamic = DynamicLeaf("attractor", dm, le, psi, xd_ca, xd_t, t_ca, beta=5.0)
    return ldynamic

def createSplineAttractor(q, qdot, x, xdot, spline, T, fk,  k=5.0):
    n = x.size(1)
    phi = fk
    dm = DiffMap("attractor", phi, q, qdot, x, xdot)
    xd_ca = ca.SX.sym("xd", n)
    psi = k * ca.norm_2(x - xd_ca)**2
    M = np.identity(n)
    le = 0.5 * ca.dot(xdot, ca.mtimes(M, xdot))
    lspline = SplineLeaf("spline_attractor", dm, le, psi, xd_ca, spline, T, beta=1.0)
    return lspline

def createSpeedAttractor(q, qdot, x, xdot, xdot_d, J, k=5.0, a_psi=10.0, a_m=0.75, m=np.array([0.3, 2.0])):
    n = x.size(1)
    phi = ca.mtimes(J, qdot) - xdot_d
    dm = DiffMap("attractor", phi, q, qdot, x, xdot)
    psi = k * (ca.norm_2(x) + 1/a_psi * ca.log(1 + ca.exp(-2*a_psi * ca.norm_2(x))))
    M = ((m[1] - m[0]) * ca.exp(-(a_m * ca.norm_2(x))**2) + m[0]) * np.identity(n)
    le = ca.dot(xdot, ca.mtimes(M, xdot))
    lforcing = ForcingLeaf("attractor", dm, le, psi)
    return lforcing

def createQuadraticAttractor(q, qdot, x, xdot, x_d, t, fk,  k=5.0, a_psi=10.0, a_m = 0.75, m=np.array([0.3, 2.0])):
    n = x.size(1)
    phi = fk - x_d
    dm = TimeVariantDiffMap("attractor", phi, q, qdot, x, xdot, t)
    psi = ca.norm_2(x)**2
    #M = ((m[1] - m[0]) * ca.exp(-(a_m * ca.norm_2(x))**2) + m[0]) * np.identity(n)
    M = np.identity(n)
    le = ca.dot(xdot, ca.mtimes(M, xdot))
    lforcing = ForcingLeaf("forcing", dm, le, psi)
    return lforcing

def createExponentialAttractor(q, qdot, x, xdot, x_d, fk,  k=5.0, a_psi=10.0, a_m = 0.75, m=np.array([0.3, 2.0])):
    n = x.size(1)
    phi = fk - x_d
    dm = DiffMap("attractor", phi, q, qdot, x, xdot)
    psi = ca.exp(0.2 * ca.norm_2(x)**2)
    #M = ((m[1] - m[0]) * ca.exp(-(a_m * ca.norm_2(x))**2) + m[0]) * np.identity(n)
    M = np.identity(n)
    le = ca.dot(xdot, ca.mtimes(M, xdot))
    damper = createDamper(x, xdot, le)
    lforcing = DampedLeaf("forcing", dm, le, psi, damper)
    return lforcing

def createRedundancySolver(q, qdot, q0, lam=0.2):
    x = ca.SX.sym("x", q.size(1))
    xdot = ca.SX.sym("xdot", qdot.size(1))
    phi = q - q0
    dm = DiffMap("redundancy", phi, q, qdot, x, xdot)
    h = ca.norm_2(xdot)**2 * (x)
    le = lam * ca.norm_2(xdot)**2
    lred = GeometryLeaf("red_res", dm, le, h)
    return lred
