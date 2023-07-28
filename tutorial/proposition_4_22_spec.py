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
    P = ca.mtimes(M, (ca.inv(M) - ca.mtimes(qdot, ca.transpose(qdot)) / ca.dot(qdot, ca.mtimes(M, qdot))))

    M_fun = ca.Function("M_" + name , [q, qdot], [M])
    f_fun = ca.Function("f_" + name, [q, qdot], [f])
    P_fun = ca.Function("P_" + name, [q, qdot], [P])
    return (L_fun, M_fun, f_fun, P_fun)

q0 = np.array([2.0, -0.5])
Lg = ca.norm_2(qdot)
(L_fun, M_fun, f_fun, P_fun) = generateLagrangian(Lg, "ex")

class Generator(object):
    def __init__(self):
        self._h = np.zeros(n)

    def update(self, q, qdot):
        self._h = h_fun(q, qdot)

    def h(self):
        return self._h

class ConservativeSpec(object):

    """Geometry as in Optimization fabrics
        M xddot + f(x, xdot) = 0
    """

    def __init__(self):
        self._f = np.zeros(n)
        self._M = np.zeros((n, n))
        self._rhs = np.zeros(n)
        self._M_aug = np.identity(2*n)
        self._rhs_aug = np.zeros(2*n)
        self._q = np.zeros(n)
        self._qdot = np.zeros(n)

    def setRHS(self):
        self._rhs = -self._f

    def augment(self):
        self._rhs_aug[0] = self._qdot[0]
        self._rhs_aug[1] = self._qdot[1]
        self._rhs_aug[2] = self._rhs[0]
        self._rhs_aug[3] = self._rhs[1]
        self._M_aug [n:, n:] = self._M

    def addNominal(self, Gen):
        P = P_fun(self._q, self._qdot)
        Gen.update(self._q, self._qdot)
        self._rhs -= np.dot(P, np.dot(self._M, Gen.h()) - self._f)

    def energy(self, q, qdot):
        return L_fun(q, qdot)

    def energies(self, z):
        q = z[:, 0:2]
        qdot = z[:, 2:4]
        energies = np.zeros(len(q))
        for i in range(len(q)):
            energies[i] = self.energy(q[i, :], qdot[i, :])
        return energies

    def updateSystem(self, z):
        self._q = z[0:n]
        self._qdot = z[n:2*n]
        self._M = M_fun(self._q, self._qdot)
        self._f = f_fun(self._q, self._qdot)

    def contDynamics(self, z, t, Gen=None):
        self.updateSystem(z)
        self.setRHS()
        if Gen:
            self.addNominal(Gen)
        self.augment()
        zdot = np.linalg.solve(self._M_aug, self._rhs_aug)
        return zdot

    def computePath(self, z0, t, Gen=None):
        sol, info = odeint(self.contDynamics, z0, t, args = (Gen,), full_output=True)
        return sol

def update(num, x1, x2, y1, y2, line1, line2, point1, point2):
    start = max(0, num - 100)
    line1.set_data(x1[start:num], y1[start:num])
    point1.set_data(x1[num], y1[num])
    line2.set_data(x2[start:num], y2[start:num])
    point2.set_data(x2[num], y2[num])
    return line1, point1, line2, point2

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
    spec = ConservativeSpec()
    gen = Generator()
    w0 = 1.0
    r0 = 2.0
    a0 = 1.0/3.0 * np.pi
    q0 = r0 * np.array([np.cos(a0), np.sin(a0)])
    q0_dot = np.array([-r0 * w0 * np.sin(a0), r0 * w0 * np.cos(a0)])
    t = np.arange(0.0, 20.00, 0.01)
    z0 = np.concatenate((q0, q0_dot))
    # solving
    sol = spec.computePath(z0, t)
    sol_en = spec.computePath(z0, t, Gen=gen)
    # plotting
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("Bending a Finsler energy")
    ax[0][0].set_title("Pure Conservative Spec")
    ax[0][1].set_title("Added Nominal Geometry")
    (x, y, line, point) = plotTraj(sol, ax[0][0], fig)
    (x2, y2, line2, point2) = plotTraj(sol_en, ax[0][1], fig)
    energies = spec.energies(sol)
    energies_en = spec.energies(sol_en)
    plotEnergies(energies, ax[1][0], t)
    plotEnergies(energies_en, ax[1][1], t)
    ani = animation.FuncAnimation(
        fig, update, len(x),
        fargs=[x, x2, y, y2, line, line2, point, point2],
        interval=25, blit=True
    )
    plt.show()


if __name__ == "__main__":
    #cProfile.run('main()', 'restats_with')
    main()