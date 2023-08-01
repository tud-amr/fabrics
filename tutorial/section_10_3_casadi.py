import pdb
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import casadi as ca
from utlis_tutorial import plotTraj, update, plotMultipleTraj

EPS = 1e-5

n = 2
q = ca.SX.sym("q", n)
qdot = ca.SX.sym("qdot", n)

h_b = np.zeros(n)
h_b_fun = ca.Function("h_b", [q, qdot], [h_b])

M_b = np.identity(n)
M_b_fun = ca.Function("M_b", [q, qdot], [M_b])

# define the forcing potential
# There is an implicit Finsler energy linked to this potential
# but I do not know how to measure it
q_d = np.array([-2.5, -3.75])
k = 5.0  # paper : 5.0
alpha_psi = 10.0
alpha_m = 0.75
m = np.array([0.3, 2.0])

# goal reaching potential
phi = q - q_d
phi_fun = ca.Function("phi", [q], [phi])
psi1 = k * (ca.norm_2(phi) + 1 / alpha_psi * ca.log(1 + ca.exp(-2 * alpha_psi * ca.norm_2(phi))))
M_psi = ((m[1] - m[0]) * ca.exp(-(alpha_m * ca.norm_2(phi)) ** 2) + m[0]) * np.identity(n)
der_psi = ca.mtimes(M_psi, ca.gradient(psi1, q))
M_psi_fun = ca.Function("M_psi", [q, qdot], [M_psi])
der_psi_fun = ca.Function("der_psi", [q, qdot], [der_psi])

# limit avoidance
q_min = np.array([-5.0, -1.0])
phi_lim = q[0] - q_min[0]
a = np.array([0.4, 0.2, 20.0, 5.0])
psi2 = a[0] / (phi_lim ** 2) + a[1] * (ca.exp(-a[2] * (phi_lim - a[3])) + 1)
lam_lim = 0.5
der_psi2 = qdot[0] ** 2 * lam_lim * ca.gradient(phi_lim, q)

# damping functions
b = np.array([0.01, 6.5])
alpha_beta = 0.5
alpha_shift = 0.5
alpha_eta = 0.5
maxExp = ca.SX.sym("maxExp", 1)
r = 1.5

beta_switch = 0.5 * (ca.tanh(-alpha_beta * (ca.norm_2(q - q_d) - r)) + 1)
energy = ca.SX.sym("energy", 1)
energy_des = ca.SX.sym("energy_des", 1)
eta_switch = 0.5 * (ca.tanh(-alpha_eta * (energy - energy_des) - alpha_shift) + 1)
beta = beta_switch * b[1] + b[0] + maxExp

beta_fun = ca.Function("beta", [q, qdot, maxExp], [beta])
eta_switch_fun = ca.Function("beta_switch", [q, qdot, energy, energy_des], [eta_switch])


def generateLagrangian(Lg, name):
    L = 0.5 * Lg ** 2
    L_fun = ca.Function("L_" + name, [q, qdot], [L])

    dL_dq = ca.gradient(L, q)
    dL_dqdot = ca.gradient(L, qdot)
    d2L_dq2 = ca.jacobian(dL_dq, q)
    d2L_dqdqdot = ca.jacobian(dL_dq, qdot)
    d2L_dqdot2 = ca.jacobian(dL_dqdot, qdot)

    M = d2L_dqdot2
    F = d2L_dqdqdot
    f_e = -dL_dq
    f = ca.mtimes(ca.transpose(F), qdot) + f_e

    M_fun = ca.Function("M_" + name, [q, qdot], [M])
    f_fun = ca.Function("f_" + name, [q, qdot], [f])
    return (L, M, f)
    # return (L_fun, M_fun, f_fun)


Lg = 0.5 * ca.norm_2(qdot)
(Lex, Mex, fex) = generateLagrangian(Lg, "ex")

Lg = 1.0 * ca.norm_2(qdot)
(Le, Me, fe) = generateLagrangian(Lg, "e")


class Energy(object):
    def __init__(self, M, f, L):
        self._M = M
        self._f = f
        self._L = L

    def L(self):
        return self._L

    def alpha(self, h, q, qdot):
        a1 = ca.dot(qdot, ca.mtimes(self._M, qdot))
        a2 = ca.dot(qdot, ca.mtimes(self._M, h) - self._f)
        alpha = a2 / a1
        return alpha


class ForcingPotential(object):
    def __init__(self, M, der_psi):
        self._dPsi = der_psi
        self._M = M

    def derPsi(self):
        return ca.mtimes(ca.inv(self._M), self._dPsi)


class SpeedController(object):
    def __init__(self, execEnergy, metricEnergy, beta_fun, eta_switch_fun):
        self._execEnergy = execEnergy
        self._metricEnergy = metricEnergy
        self._b = np.array([0.01, 6.5])
        a_beta = 0.5
        a_shift = 0.5
        a_eta = 0.5
        r = 1.5
        self._eta = 0.5 * (ca.tanh(-a_eta * (metricEnergy.L() - execEnergy.L()) - a_shift) + 1)
        self._beta_switch = 0.5 * (ca.tanh(-a_beta * (ca.norm_2(q - q_d) - r)) + 1)

    def alpha_ex(self, h, derPsi, q, qdot):
        self._a_ex0 = self._execEnergy.alpha(h, q, qdot)
        self._a_expsi = self._execEnergy.alpha(h + derPsi, q, qdot)
        self._alpha_le = self._metricEnergy.alpha(h, q, qdot)
        self._alpha_ex = self._eta * self._a_ex0 + (1 - self._eta) * self._a_expsi
        return self._alpha_ex

    def beta(self, q, qdot):
        beta = self._beta_switch * b[1] + b[0] + ca.fmax(0.0, self._alpha_ex - self._alpha_le)
        return beta


class Geometry(object):

    def __init__(self, h, Le=None, Forcing=None, SpeedController=None):
        self._rhs = ca.vertcat(0.0, 0.0)
        self._q = q
        self._qdot = qdot
        self._z = ca.vertcat(q, qdot)
        self._h = h
        self._forcing = Forcing
        self._derPsi = ca.vertcat(0.0, 0.0)
        self._le = Le
        self._speedController = SpeedController
        if SpeedController:
            self._le = None

    def setRHS(self):
        self._rhs = -self._h

    def augment(self):
        self._rhs_aug = ca.vertcat(self._z[2], self._z[3], self._rhs[0], self._rhs[1])

    def force(self):
        self._derPsi = self._forcing.derPsi()
        self._rhs -= self._derPsi

    def speedControl(self):
        alpha_ex = self._speedController.alpha_ex(self._h, self._derPsi, self._q, self._qdot)
        beta = self._speedController.beta(self._q, self._qdot)
        self._rhs += alpha_ex * self._qdot
        self._rhs -= beta * self._qdot

    def energize(self):
        alpha = self._le.alpha(self._h + self._derPsi, self._q, self._qdot)
        self._rhs += alpha * self._qdot

    def createSolver(self, dt=0.01):
        self._dt = dt
        self.setRHS()
        if self._forcing:
            self.force()
        if self._speedController:
            self.speedControl()
        if self._le:
            self.energize()
        self.augment()
        ode = {}
        ode['x'] = self._z
        ode['ode'] = self._rhs_aug
        self._int_fun = ca.integrator("int_fun", 'idas', ode, {'tf': dt})

    def computePath(self, z0, T):
        num_steps = int(T / self._dt)
        z = z0
        sols = np.zeros((num_steps, 4))
        max_step = num_steps
        for i in range(num_steps):
            if np.linalg.norm(z) < 0.1:
                break
            try:
                res = self._int_fun(x0=z)
            except Exception as e:
                if i < max_step:
                    max_step = i
                break
            z = np.array(res['xf'])[:, 0]
            qdot = z[n:2 * n]
            qdot_norm = np.linalg.norm(qdot)
            sols[i, :] = z
            if qdot_norm < 0.030:
                if i < max_step:
                    max_step = i
                print("zero velocity")
                break
        print("finished")
        return sols[:max_step, :]


def main():
    # setup 
    lex = Energy(Mex, fex, Lex)
    le = Energy(Me, fe, Le)
    forcing = ForcingPotential(M_b, der_psi)
    limit_forcing = ForcingPotential(M_b, der_psi2)
    speedController = SpeedController(lex, le, beta, eta_switch)
    geo1 = Geometry(h=h_b, Forcing=forcing)
    geo1.createSolver(dt=0.01)
    geo2 = Geometry(h=h_b, Forcing=forcing, SpeedController=speedController)
    geo2.createSolver(dt=0.01)
    # geo3 = Geometry(h_fun = h_b_fun, Forcing=forcing, SpeedController=speedController)
    geos = [geo1, geo2]
    q0 = np.array([2.0, 3.0])
    v0 = 1.5
    a0s = [(i * np.pi) / 7 for i in range(14)]
    T = 20.0
    # solving
    sols = []
    for geo in geos:
        geoSols = []
        for a0 in a0s:
            print(a0)
            if a0 == 0.0:
                continue
            q0_dot = v0 * np.array([np.cos(a0), np.sin(a0)])
            print(q0)
            z0 = np.concatenate((q0, q0_dot))
            print("Compute path for a0 : ", a0)
            geoSols.append(geo.computePath(z0, T))
        sols.append(geoSols)
    sol1 = sols[0][0]
    sol2 = sols[1][0]
    sol3 = sols[1][0]
    # plotting
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Example for goal attraction")
    ax[0].set_title("No Damping")
    ax[1].set_title("With Damping")
    plotMultipleTraj(sols[0], ax[0], fig, int(T / 0.01 / 25))
    plotMultipleTraj(sols[1], ax[1], fig, int(T / 0.01 / 25))
    (x, y, line, point) = plotTraj(sol1, ax[0], fig)
    (x2, y2, line2, point2) = plotTraj(sol2, ax[1], fig)
    animation_data = [
        [line, line2],
        [point, point2],
        [{'x': x, 'y': y}, {'x': x2, 'y': y2}]
    ]
    ani = animation.FuncAnimation(
        fig, update, len(x),
        fargs=animation_data,
        interval=25, blit=True
    )
    plt.show()


if __name__ == "__main__":
    # cProfile.run('main()', 'restats_with')
    main()
