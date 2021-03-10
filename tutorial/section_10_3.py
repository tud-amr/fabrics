import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import casadi as ca

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
k = 5.0 # paper : 5.0
alpha_psi = 10.0
alpha_m = 0.75
m = np.array([0.3, 2.0])

phi = q - q_d
phi_fun = ca.Function("phi", [q], [phi])
psi1 = k * (ca.norm_2(phi) + 1/alpha_psi * ca.log(1 + ca.exp(-2*alpha_psi * ca.norm_2(phi))))
M_psi = ((m[1] - m[0]) * ca.exp(-(alpha_m * ca.norm_2(phi))**2) + m[0]) * np.identity(n)
der_psi = ca.mtimes(M_psi, ca.gradient(psi1, q))
M_psi_fun = ca.Function("M_psi", [q, qdot], [M_psi])
der_psi_fun = ca.Function("der_psi", [q, qdot], [der_psi])

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

    M_fun = ca.Function("M_" + name , [q, qdot], [M])
    f_fun = ca.Function("f_" + name, [q, qdot], [f])
    return (L_fun, M_fun, f_fun)

Lg = 0.5 * ca.norm_2(qdot)
(Lex, Mex, fex) = generateLagrangian(Lg, "ex")

Lg = 1.0 * ca.norm_2(qdot)
(Le, Me, fe) = generateLagrangian(Lg, "e")


class Energy(object):
    def __init__(self, M_fun, f_fun, L_fun):
        self._Me = np.zeros((n, n))
        self._fe = np.zeros(2)
        self.M_fun = M_fun
        self.f_fun = f_fun
        self.L_fun = L_fun

    def energy(self, q, qdot):
        return self.L_fun(q, qdot)

    def energies(self, z):
        q = z[:, 0:2]
        qdot = z[:, 2:4]
        energies = np.zeros(len(q))
        for i in range(len(q)):
            energies[i] = self.energy(q[i, :], qdot[i, :])
        return energies

    def update(self, q, qdot):
        self._Me = self.M_fun(q, qdot)
        self._fe = self.f_fun(q, qdot)

    def alpha(self, h, q, qdot):
        self.update(q, qdot)
        a1 = np.dot(qdot, np.dot(self._Me, qdot))
        a2 = np.dot(qdot, np.dot(self._Me, h) - self._fe)
        alpha = a2/a1
        return alpha

class ForcingPotential(object):
    def __init__(self, M_fun, der_psi_fun):
        self._dPsi = np.zeros(n)
        self._M = np.identity(n)
        self.M_fun = M_fun
        self.der_psi_fun = der_psi_fun

    def update(self, q, qdot):
        self._dPsi = self.der_psi_fun(q, qdot)
        self._M = self.M_fun(q, qdot)

    def derPsi(self):
        return np.linalg.solve(self._M, self._dPsi)[:, 0]

class SpeedController(object):
    def __init__(self, execEnergy, metricEnergy, beta_fun, eta_switch_fun):
        self._execEnergy = execEnergy
        self._metricEnergy = metricEnergy
        self.beta_fun = beta_fun
        self.eta_switch_fun = eta_switch_fun
        self._alpha_ex = 0.0

    def alpha_ex(self, h, derPsi, q, qdot):
        self._execEnergy.update(q, qdot)
        self._metricEnergy.update(q, qdot)
        self._a_ex0 = self._execEnergy.alpha(h, q, qdot)
        self._a_expsi = self._execEnergy.alpha(h + derPsi, q, qdot)
        self._a_le = self._metricEnergy.alpha(h, q, qdot)
        e1 = self._metricEnergy.energy(q, qdot)
        e2 = self._execEnergy.energy(q, qdot)
        eta = self.eta_switch_fun(q, qdot, e1, e2)
        self._alpha_ex = eta * self._a_ex0 +  (1 - eta) * self._a_expsi
        return self._alpha_ex

    def beta(self, q, qdot):
        beta = self.beta_fun(q, qdot, max(0.0, self._alpha_ex - self._a_le))
        return beta

class Geometry(object):
    """Geometry as in Optimization fabrics
        xddot + h(x, xdot) = 0
    """

    def __init__(self, h_fun, Le = None, Forcing=None, SpeedController=None):
        self._n = 2
        self.h_fun = h_fun
        self._h = np.zeros(n)
        self._rhs = np.zeros(n)
        self._rhs_aug = np.zeros(2*n)
        self._q = np.zeros(n)
        self._qdot = np.zeros(n)
        self._forcing = Forcing
        self._derPsi = np.zeros(n)
        self._le = Le
        self._speedController = SpeedController
        if SpeedController:
            self._le = None

    def setRHS(self):
        self._h = self.h_fun(self._q, self._qdot)
        self._rhs = np.zeros(n)
        self._rhs -= self._h

    def augment(self):
        self._rhs_aug[0] = self._qdot[0]
        self._rhs_aug[1] = self._qdot[1]
        self._rhs_aug[2] = self._rhs[0]
        self._rhs_aug[3] = self._rhs[1]

    def force(self):
        self._forcing.update(self._q, self._qdot)
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

    def contDynamics(self, z, t):
        self._q = z[0:n]
        self._qdot = z[n:2*n]
        self.setRHS()
        if self._forcing:
            self.force()
        if self._speedController:
            self.speedControl()
        if self._le:
            self.energize()
        self.augment()
        zdot = self._rhs_aug
        return zdot

    def computePath(self, z0, t):
        sol, info = odeint(self.contDynamics, z0, t, full_output=True)
        return sol

def update(num, x1, x2, x3, y1, y2, y3, line1, line2, line3,  point1, point2, point3):
    start = max(0, num - 100)
    line1.set_data(x1[start:num], y1[start:num])
    point1.set_data(x1[num], y1[num])
    line2.set_data(x2[start:num], y2[start:num])
    point2.set_data(x2[num], y2[num])
    line3.set_data(x3[start:num], y3[start:num])
    point3.set_data(x3[num], y3[num])
    return line1, point1, line2, point2, line3, point3

def plotTraj(sol, ax, fig):
    x = sol[:, 0]
    y = sol[:, 1]
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.plot(x, y)
    (line,) = ax.plot(x, y, color="k")
    (point,) = ax.plot(x, y, "rx")
    return (x, y, line, point)

def plotMultipleTraj(sols, ax, fig):
    for sol in sols:
        x = sol[:, 0]
        y = sol[:, 1]
        ax.set_xlim([-4, 4])
        ax.set_ylim([-4, 4])
        ax.plot(x, y)

def plotEnergies(energies, ax, t):
    ax.plot(t, energies)

def main():
    # setup 
    lex = Energy(Mex, fex, Lex)
    le = Energy(Me, fe, Le)
    forcing = ForcingPotential(M_b_fun, der_psi_fun)
    speedController = SpeedController(lex, le, beta_fun, eta_switch_fun)
    geo1 = Geometry(h_fun = h_b_fun)
    geo3 = Geometry(h_fun = h_b_fun, Forcing=forcing, SpeedController=speedController)
    geos = [geo1, geo3]
    q0 = np.array([2.0, 3.0])
    v0 = 1.5
    a0s = [((i * np.pi)/7) for i in range(14)]
    t = np.arange(0.0, 16.00, 0.001)
    # solving
    # solving
    sols = []
    for geo in geos:
        geoSols = []
        for a0 in a0s:
            if a0 == 0.0:
                continue
            q0_dot = v0 * np.array([np.cos(a0), np.sin(a0)])
            z0 = np.concatenate((q0, q0_dot))
            print("Compute path for a0 : ", a0)
            geoSols.append(geo.computePath(z0, t))
        sols.append(geoSols)
    sol1 = sols[0][0]
    sol2 = sols[1][0]
    sol3 = sol2
    # plotting
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("<Title>")
    ax[0][0].set_title("<Subtitle>")
    ax[0][1].set_title("<Subtitle>")
    ax[1][0].set_title("<Subtitle>")
    plotMultipleTraj(sols[0], ax[0][0], fig)
    plotMultipleTraj(sols[1], ax[0][1], fig)
    (x, y, line, point) = plotTraj(sol1, ax[0][0], fig)
    (x2, y2, line2, point2) = plotTraj(sol2, ax[0][1], fig)
    (x3, y3, line3, point3) = plotTraj(sol3, ax[1][0], fig)
    ani = animation.FuncAnimation(
        fig, update, len(x),
        fargs=[x, x2, x3, y, y2, y3, line, line2, line3, point, point2, point3],
        interval=10, blit=True
    )
    plt.show()


if __name__ == "__main__":
    #cProfile.run('main()', 'restats_with')
    main()
