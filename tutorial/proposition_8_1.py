import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import casadi as ca

n = 2
q = ca.SX.sym("q", n)
q_d = ca.SX.sym("q_d", n)
qdot = ca.SX.sym("qdot", n)

r = ca.norm_2(q)
a = ca.arctan2(q[1], q[0])
w = ca.norm_2(qdot)/ca.norm_2(q)
h = ca.SX.sym("h", 2)
b = r
h[0] = b * r * w**2 * ca.cos(a)
h[1] = b * r * w**2 * ca.sin(a)
h_fun = ca.Function("h", [q, qdot], [h])


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

Lg = ca.norm_2(qdot)
(Lex, Mex, fex) = generateLagrangian(Lg, "ex")

q0 = np.array([2.0, -0.5])
Lg = ca.norm_2(qdot) * 1/ca.norm_2(q - q0)**2
(Le, Me, fe) = generateLagrangian(Lg, "e")

# forcing potential
qd = np.array([-2, -2])
psi = ca.norm_2(q - qd) * ca.norm_2(qdot)
psi = 0.07 * ca.norm_2(q - qd)
der_psi = ca.gradient(psi, q)
der_psi_fun = ca.Function("der_psi", [q, qdot], [der_psi])

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
    def __init__(self, M_fun):
        self._dPsi = np.zeros(n)
        self._M = np.identity(n)
        self.M_fun = M_fun

    def update(self, q, qdot):
        self._dPsi = der_psi_fun(q, qdot)
        self._M = self.M_fun(q, qdot)

    def derPsi(self):
        return np.linalg.solve(self._M, self._dPsi)[:, 0]

class SpeedController(object):
    def __init__(self, execEnergy, metricEnergy):
        self._execEnergy = execEnergy
        self._metricEnergy = metricEnergy
        self._beta = 0.0
        self._alpha_ex = 0.0

    def alpha_ex(self, h, derPsi, q, qdot):
        self._execEnergy.update(q, qdot)
        self._metricEnergy.update(q, qdot)
        self._a_ex0 = self._execEnergy.alpha(h, q, qdot)
        self._a_expsi = self._execEnergy.alpha(h + derPsi, q, qdot)
        self._a_le = self._metricEnergy.alpha(h, q, qdot)
        # interpolation
        eta = 0.050
        self._alpha_ex = eta * self._a_ex0 +  (1 - eta) * self._a_expsi
        return self._alpha_ex

    def beta(self):
        return max(0.0, 1.05 * (self._alpha_ex - self._a_le))

class Geometry(object):

    """Geometry as in Optimization fabrics
        xddot + h(x, xdot) = 0
    """

    def __init__(self, Le = None, Forcing=None, SpeedController=None):
        self._n = 2
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
        self._h = h_fun(self._q, self._qdot)
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
        beta = self._speedController.beta()
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

def plotEnergies(energies, ax, t):
    ax.plot(t, energies)

def main():
    # setup 
    lex = Energy(Mex, fex, Lex)
    le = Energy(Me, fe, Le)
    forcing = ForcingPotential(Me)
    speedController = SpeedController(lex, le)
    geo1 = Geometry(Le=le)
    geo2 = Geometry(Le=lex)
    #geo3 = Geometry(Le=lex, Forcing=forcing)
    geo3 = Geometry(Forcing=forcing, SpeedController=speedController)
    w0 = 1.0
    r0 = 2.0
    a0 = 1.0/3.0 * np.pi
    q0 = r0 * np.array([np.cos(a0), np.sin(a0)])
    q0_dot = np.array([-r0 * w0 * np.sin(a0), r0 * w0 * np.cos(a0)])
    t = np.arange(0.0, 20.00, 0.01)
    z0 = np.concatenate((q0, q0_dot))
    # solving
    print("Computing energized with Le")
    sol1 = geo1.computePath(z0, t)
    print("Computing energized with Lex")
    sol2 = geo2.computePath(z0, t)
    print("Computing forced and speed controlled")
    sol3 = geo3.computePath(z0, t)
    # plotting
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Speed Control through forcing")
    ax[0][0].set_title("Constant Finsler energy")
    ax[0][1].set_title("Constant kinetic energy")
    ax[0][2].set_title("Speed controlled")
    (x, y, line, point) = plotTraj(sol1, ax[0][0], fig)
    (x2, y2, line2, point2) = plotTraj(sol2, ax[0][1], fig)
    (x3, y3, line3, point3) = plotTraj(sol3, ax[0][2], fig)
    ax[0][2].plot(qd[0], qd[1], 'go')
    energies1_e = le.energies(sol1)
    energies2_e = le.energies(sol2)
    energies3_e = le.energies(sol3)
    energies1_ex = lex.energies(sol1)
    energies2_ex = lex.energies(sol2)
    energies3_ex = lex.energies(sol3)
    plotEnergies(energies1_e, ax[1][0], t)
    plotEnergies(energies2_e, ax[1][1], t)
    plotEnergies(energies3_e, ax[1][2], t)
    plotEnergies(energies1_ex, ax[1][0], t)
    plotEnergies(energies2_ex, ax[1][1], t)
    plotEnergies(energies3_ex, ax[1][2], t)
    ax[1][0].legend(["Le", "Le"])
    ax[1][1].legend(["Le", "Le"])
    ax[1][2].legend(["Le", "Le"])
    ani = animation.FuncAnimation(
        fig, update, len(x),
        fargs=[x, x2, x3, y, y2, y3, line, line2, line3, point, point2, point3],
        interval=10, blit=True
    )
    plt.show()


if __name__ == "__main__":
    #cProfile.run('main()', 'restats_with')
    main()
