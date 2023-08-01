import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import casadi as ca
from utlis_tutorial import plotTraj, plotEnergies, update

n = 2
q = ca.SX.sym("q", n)
q_d = ca.SX.sym("q_d", n)
qdot = ca.SX.sym("qdot", n)

r = ca.norm_2(q)
a = ca.arctan2(q[1], q[0])
w = ca.norm_2(qdot) / ca.norm_2(q)
h = ca.SX.sym("h", 2)
b = r
h[0] = b * r * w ** 2 * ca.cos(a)
h[1] = b * r * w ** 2 * ca.sin(a)
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

    M_fun = ca.Function("M_" + name, [q, qdot], [M])
    f_fun = ca.Function("f_" + name, [q, qdot], [f])
    return (L_fun, M_fun, f_fun)


q0 = np.array([2.0, -0.5])
Lg = ca.norm_2(qdot)
(L_fun, M_fun, f_fun) = generateLagrangian(Lg, "ex")


class EnergyLagrangian(object):

    def __init__(self):
        self._Me = np.zeros((n, n))
        self._fe = np.zeros(2)

    def energy(self, q, qdot):
        return L_fun(q, qdot)

    def energies(self, z):
        q = z[:, 0:2]
        qdot = z[:, 2:4]
        energies = np.zeros(len(q))
        for i in range(len(q)):
            energies[i] = self.energy(q[i, :], qdot[i, :])
        return energies

    def update(self, q, qdot):
        self._Me = M_fun(q, qdot)
        self._fe = f_fun(q, qdot)

    def alpha(self, h, q, qdot):
        self.update(q, qdot)
        a1 = np.dot(qdot, np.dot(self._Me, qdot))
        a2 = np.dot(qdot, np.dot(self._Me, h) - self._fe)
        alpha = -a2 / a1
        return alpha * qdot


class Geometry(object):
    """Geometry as in Optimization fabrics
        xddot + h(x, xdot) = 0
    """

    def __init__(self):
        self._n = 2
        self._h = np.zeros(n)
        self._rhs = np.zeros(n)
        self._rhs_aug = np.zeros(2 * n)
        self._q = np.zeros(n)
        self._qdot = np.zeros(n)

    def setRHS(self):
        self._h = h_fun(self._q, self._qdot)
        self._rhs = -self._h

    def augment(self):
        self._rhs_aug[0] = self._qdot[0]
        self._rhs_aug[1] = self._qdot[1]
        self._rhs_aug[2] = self._rhs[0]
        self._rhs_aug[3] = self._rhs[1]

    def energize(self, Le):
        h_ene = Le.alpha(self._h, self._q, self._qdot)
        self._rhs -= Le.alpha(self._h, self._q, self._qdot)

    def contDynamics(self, z, t, Le=None):
        self._q = z[0:n]
        self._qdot = z[n:2 * n]
        self.setRHS()
        if Le:
            self.energize(Le)
        self.augment()
        zdot = self._rhs_aug
        return zdot

    def computePath(self, z0, t, Le=None):
        if Le:
            sol, info = odeint(self.contDynamics, z0, t, args=(Le,), full_output=True)
        else:
            sol, info = odeint(self.contDynamics, z0, t, full_output=True)
        return sol


def main():
    # setup 
    geo = Geometry()
    Le = EnergyLagrangian()
    w0 = 1.0
    r0 = 2.0
    a0 = 1.0 / 3.0 * np.pi
    q0 = r0 * np.array([np.cos(a0), np.sin(a0)])
    q0_dot = np.array([-r0 * w0 * np.sin(a0), r0 * w0 * np.cos(a0)])
    t = np.arange(0.0, 20.00, 0.01)
    z0 = np.concatenate((q0, q0_dot))
    # solving
    sol = geo.computePath(z0, t)
    sol_en = geo.computePath(z0, t, Le)
    # plotting
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("Energization a geometry")
    ax[0][0].set_title("Original geometry generator")
    ax[0][1].set_title("Energized geometry generator")
    (x, y, line, point) = plotTraj(sol, ax[0][0], fig)
    (x2, y2, line2, point2) = plotTraj(sol_en, ax[0][1], fig)
    energies = Le.energies(sol)
    energies_en = Le.energies(sol_en)
    plotEnergies(energies, ax[1][0], t)
    plotEnergies(energies_en, ax[1][1], t)
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
