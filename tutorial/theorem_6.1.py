import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import casadi as ca

""" pull back of energized, generic generator
PAY ATTENTION TO THE DEFINITION OF PHI
"""

n = 2
q = ca.SX.sym("q", n)
qdot = ca.SX.sym("qdot", n)
x = ca.SX.sym("x", n)
xdot = ca.SX.sym("xdot", n)

# differential map and jacobian phi : Q -> X
phi = ca.SX.sym("phi", 2)
phi[0] = q[0] * ca.cos(q[1])
phi[1] = q[0] * ca.sin(q[1])
phi_fun = ca.Function("phi", [q], [phi])
J = ca.jacobian(phi, q)
Jdot = ca.jacobian(ca.mtimes(J, qdot), q)
J_fun = ca.Function("J", [q], [J])
Jdot_fun = ca.Function("Jdot", [q, qdot], [Jdot])

# inverse 
phi_inv = ca.SX.sym("phi_inv", 2)
phi_inv[0] = ca.norm_2(x)
phi_inv[1] = ca.atan2(x[1], x[0])

phi_inv_fun = ca.Function("phi_inv", [x], [phi_inv])
J_inv = ca.jacobian(phi_inv, x)
J_inv_fun = ca.Function("J_inv", [x], [J_inv])


def generateLagrangian(Lg, name):
    L = 0.5 * Lg ** 2
    L_fun = ca.Function("L_" + name, [x, xdot], [L])

    dL_dx = ca.gradient(L, x)
    dL_dxdot = ca.gradient(L, xdot)
    d2L_dx2 = ca.jacobian(dL_dx, x)
    d2L_dxdxdot = ca.jacobian(dL_dx, xdot)
    d2L_dxdot2 = ca.jacobian(dL_dxdot, xdot)

    M = d2L_dxdot2
    F = d2L_dxdxdot
    f_e = -dL_dx
    f = ca.mtimes(ca.transpose(F), xdot) + f_e
    P = ca.mtimes(M, (ca.inv(M) - ca.mtimes(xdot, ca.transpose(xdot)) / ca.dot(xdot, ca.mtimes(M, xdot))))

    M_fun = ca.Function("M_" + name , [x, xdot], [M])
    f_fun = ca.Function("f_" + name, [x, xdot], [f])
    P_fun = ca.Function("P_" + name, [x, xdot], [P])
    return (L_fun, M_fun, f_fun, P_fun)

x0 = np.array([-2.00, -0.0])
L_g = 10.0* ca.norm_2(xdot) * ca.norm_2(x - x0)
L_g = ca.norm_2(xdot)
(L_fun, M_fun, f_fun, P_fun) = generateLagrangian(L_g, "g")

r = ca.norm_2(x)
a = ca.arctan2(x[1], x[0])
w = ca.norm_2(xdot)/ca.norm_2(x)
h = ca.SX.sym("h", 2)
b = r
h[0] = b * r * w**2 * ca.cos(a)
h[1] = b * r * w**2 * ca.sin(a)
h_fun = ca.Function("h", [x, xdot], [h])

class Pulling(object):
    def __init__(self):
        self._J = np.zeros((n, n))
        self._Jdot = np.zeros((n, n))
        self._Jt = np.zeros((n, n))
        self._qdot = np.zeros(n)

    def x2q(self, zx):
        x = zx[0:2]
        xdot = zx[2:4]
        q = np.array(phi_inv_fun(x))[:, 0]
        qdot = np.dot(J_inv_fun(x), xdot)
        return (q, qdot)

    def q2x(self, zq):
        q = zq[0:2]
        qdot = zq[2:4]
        zTrans = np.zeros(4)
        zTrans[0:2] = np.array(phi_fun(q))[:, 0]
        J = np.array(J_fun(q))
        zTrans[2:4] = np.dot(J, qdot)
        return zTrans

    def qs2xs(self, zqs):
        n = zqs[:, 0].size
        zTrans = np.zeros((n, 4))
        for i in range(n):
            zTrans[i, :] = self.q2x(zqs[i, :])
        return zTrans

    def update(self, q, qdot):
        self._J = J_fun(q)
        self._Jt = np.transpose(self._J)
        self._Jdot = Jdot_fun(q, qdot)
        self._qdot = qdot

    def J(self):
        return self._J

    def M(self, M):
        M_pulled = np.dot(self._Jt, np.dot(M, self._J))
        return M_pulled

    def f(self, f):
        return np.dot(self._Jt, f)

    def f_add(self, M):
        f2 = np.dot(np.dot(self._Jt, M), np.dot(self._Jdot, self._qdot))
        return f2

class EnergyLagrangian(object):

    def __init__(self, Pulling=None):
        self._pulling = Pulling
        if Pulling:
            self._M_or = np.zeros((n, n))
            self._x_or = np.zeros(n)
            self._xdot_or = np.zeros(n)
        self._Me = np.zeros((n, n))
        self._fe = np.zeros(2)

    def energy(self, x, xdot):
        return L_fun(x, xdot)

    def energies(self, z):
        x = z[:, 0:2]
        xdot = z[:, 2:4]
        energies = np.zeros(len(x))
        for i in range(len(x)):
            energies[i] = self.energy(x[i, :], xdot[i, :])
        return energies

    def update(self, x, xdot):
        if self._pulling:
            self._pulling.update(x, xdot)
            z_or = self._pulling.q2x(np.concatenate((x, xdot)))
            self._x_or = z_or[0:2]
            self._xdot_or = z_or[2:4]
            self._M_or = M_fun(self._x_or, self._xdot_or)
            self._f_or = f_fun(self._x_or, self._xdot_or)
            self._Me = self._pulling.M(self._M_or)
            self._fe = self._pulling.f(self._f_or)[:, 0]
        else:
            self._Me = M_fun(x, xdot)
            self._fe = f_fun(x, xdot)

    def alpha(self, h, x, xdot):
        self.update(x, xdot)
        if self._pulling:
            a1 = np.dot(xdot, np.dot(self._Me, xdot))
            a2 = np.dot(xdot, h - self._fe)
            alpha = -a2/a1
            J = self._pulling.J()
            return alpha * np.dot(np.dot(np.transpose(J), J), xdot)
        else:
            a1 = np.dot(xdot, np.dot(self._Me, xdot))
            a2 = np.dot(xdot, np.dot(self._Me, h) - self._fe)
            alpha = -a2/a1
            return alpha * xdot

class Spec(object):

    """Spec as in Optimization fabrics
        M xddot + f(x, xdot) = 0
        In this specific case f = M h
        h is the generator and M the energy metric
    """

    def __init__(self, Pulling=None, first="energy"):
        self._first = first
        self._pulling = Pulling
        if Pulling:
            self._x_or = np.zeros(n)
            self._xdot_or = np.zeros(n)
            self._M_or = np.zeros((n, n))
            self._f_or = np.zeros(n)
        self._f = np.zeros(n)
        self._M = np.zeros((n, n))
        self._rhs = np.zeros(n)
        self._M_aug = np.identity(2*n)
        self._rhs_aug = np.zeros(2*n)
        self._x = np.zeros(n)
        self._xdot = np.zeros(n)

    def emptyRHS(self):
        self._rhs = np.zeros(n)

    def augment(self):
        self._rhs_aug[0] = self._xdot[0]
        self._rhs_aug[1] = self._xdot[1]
        self._rhs_aug[2] = self._rhs[0]
        self._rhs_aug[3] = self._rhs[1]
        self._M_aug [n:, n:] = self._M

    def addNominal(self, Gen):
        P = P_fun(self._x, self._xdot)
        Gen.update(self._x, self._xdot)
        self._rhs -= np.dot(P, np.dot(self._M, Gen.h()) - self._f)

    def energize(self, Le):
        if self._pulling and self._first == "energy":
            f_en = Le.alpha(self._f_or, self._x_or, self._xdot_or)
            a = self._pulling.f(f_en)
            self._rhs -= self._pulling.f(f_en)
        elif self._pulling and self._first == "pull":
            a = self._pulling.f(Le.alpha(self._pulling.f(self._f_or), self._x, self._xdot))
            a = Le.alpha(self._pulling.f(self._f_or), self._x, self._xdot)
            self._rhs -= a
        else:
            self._rhs -= np.dot(self._M, Le.alpha(self._f, self._x, self._xdot))

    def updateSystem(self, z):
        self._x = z[0:n]
        self._xdot = z[n:2*n]
        if self._pulling:
            # Update pulling
            self._pulling.update(self._x, self._xdot)
            z_or = self._pulling.q2x(z)
            self._x_or = z_or[0:2]
            self._xdot_or = z_or[2:4]
            self._M_or = M_fun(self._x_or, self._xdot_or)
            h_or = h_fun(self._x_or, self._xdot_or)
            self._f_or = np.dot(self._M_or, h_or)[:, 0]
            self._M = self._pulling.M(self._M_or)
            self._rhs -= self._pulling.f(self._f_or) + self._pulling.f_add(self._M_or)
        else:
            self._M = M_fun(self._x, self._xdot)
            h = h_fun(self._x, self._xdot)
            self._f = np.dot(self._M, h)[:, 0]
            self._rhs -= self._f

    def contDynamics(self, z, t, Le=None):
        self.emptyRHS()
        self.updateSystem(z)
        if Le:
            self.energize(Le)
        self.augment()
        zdot = np.linalg.solve(self._M_aug, self._rhs_aug)
        return zdot

    def computePath(self, z0, t, Le=None):
        sol, info = odeint(self.contDynamics, z0, t, args = (Le,), full_output=True)
        return sol

def update(num, x, x1, x2, y, y1, y2, line, line1, line2, point, point1, point2):
    start = max(0, num - 100)
    line.set_data(x[start:num], y[start:num])
    point.set_data(x[num], y[num])
    line1.set_data(x1[start:num], y1[start:num])
    point1.set_data(x1[num], y1[num])
    line2.set_data(x2[start:num], y2[start:num])
    point2.set_data(x2[num], y2[num])
    return line, point, line1, point1, line2, point2

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
    spec = Spec()
    pull = Pulling()
    le = EnergyLagrangian()
    le_pull = EnergyLagrangian(Pulling=pull)
    spec_pull = Spec(Pulling=pull)
    spec_pull_order = Spec(Pulling=pull, first="pull")
    w0 = 1.0
    r0 = 2.0
    a0 = 1.5/3.0 * np.pi
    x0 = r0 * np.array([np.cos(a0), np.sin(a0)])
    x0_dot = np.array([-r0 * w0 * np.sin(a0), r0 * w0 * np.cos(a0)])
    t = np.arange(0.0, 10.02, 0.01)
    z0 = np.concatenate((x0, x0_dot))
    x0_pull = np.array([r0, a0])
    x0_dot_pull = np.array([0.0, w0])
    z0_pull = np.concatenate((x0_pull, x0_dot_pull))
    # solving
    print("Not pulled")
    sol = spec.computePath(z0, t, Le=le)
    print("energize + pull")
    sol_p = spec_pull.computePath(z0_pull, t, Le=le)
    sol_p_t = pull.qs2xs(sol_p)
    print("pull + energize")
    sol_p2 = spec_pull_order.computePath(z0_pull, t, Le=le_pull)
    sol_p2_t = pull.qs2xs(sol_p2)
    # plotting
    fig, ax = plt.subplots(2, 3, figsize=(10, 10))
    fig.suptitle("Commuting energization : Cartesian vs Polar")
    ax[0][0].set_title("Energized")
    ax[0][1].set_title("Energized and pulled")
    ax[0][2].set_title("Pulled and energized")
    (x, y, line, point) = plotTraj(sol, ax[0][0], fig)
    (x2, y2, line2, point2) = plotTraj(sol_p_t, ax[0][1], fig)
    (x3, y3, line3, point3) = plotTraj(sol_p2_t, ax[0][2], fig)
    energies = le.energies(sol)
    energies_p = le.energies(sol_p_t)
    energies_p2 = le.energies(sol_p2_t)
    plotEnergies(energies, ax[1][0], t)
    plotEnergies(energies_p, ax[1][1], t)
    plotEnergies(energies_p, ax[1][2], t)
    ani = animation.FuncAnimation(
        fig, update, len(x),
        fargs=[x, x2, x3, y, y2, y3, line, line2, line3, point, point2, point3],
        interval=25, blit=True
    )
    plt.show()


if __name__ == "__main__":
    main()
