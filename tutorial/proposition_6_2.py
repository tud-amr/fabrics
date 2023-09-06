import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import casadi as ca
from tutorial_utils import plot_polar_trajectory, update

n = 2
m = 2
q = ca.SX.sym("q", m)
q_d = ca.SX.sym("q_d", m)
qdot = ca.SX.sym("qdot", m)

x = ca.SX.sym("x", n)
xdot = ca.SX.sym("xdot", n)

# differential map and jacobian phi : Q -> X
phi1 = ca.SX.sym("phi1", 2)
phi1[0] = q[0] * ca.cos(q[1])
phi1[1] = q[0] * ca.sin(q[1])
phi1_fun = ca.Function("phi1", [q], [phi1])
J1 = ca.jacobian(phi1, q)
J1dot = ca.jacobian(ca.mtimes(J1, qdot), q)
J1_fun = ca.Function("J1", [q], [J1])
J1dot_fun = ca.Function("Jdot1", [q, qdot], [J1dot])

r = ca.norm_2(x)
a = ca.arctan2(x[1], x[0])
w = ca.norm_2(xdot) / ca.norm_2(x)
h1 = ca.SX.sym("h1", 2)
b = r ** 1
h1[0] = b * r * w ** 2 * ca.cos(a)
h1[1] = b * r * w ** 2 * ca.sin(a)
h1_fun = ca.Function("h1", [x, xdot], [h1])

k = 0.5
lam = 0.3
r_obst = 1.00
x_obst = np.array([2.0, 0.0])
psi = k * r_obst ** 2 / ((ca.norm_2(x - x_obst) - r_obst) ** 2)
derPsi = ca.gradient(psi, x)
h2 = lam * (ca.norm_2(xdot)) ** 2 * derPsi
h2_fun = ca.Function('h2', [x, xdot], [h2])


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
Lg = ca.norm_2(qdot) * 1 / ca.norm_2(q - q0) ** 2
# Lg = ca.norm_2(qdot)
(L1_fun, M1_fun, f1_fun) = generateLagrangian(Lg, "ex")


class Geometry(object):

    def __init__(self, h_fun):
        self.h_fun = h_fun

    def h(self, x, xdot):
        return self.h_fun(x, xdot)


class Leaf(object):
    def __init__(self, M_fun, phi_fun, J_fun, J_dot_fun, geo):
        self._geo = geo
        self.M_fun = M_fun
        self.phi_fun = phi_fun
        self.J_fun = J_fun
        self.J_dot_fun = J_dot_fun
        self._M = np.zeros((n, n))
        self._x = np.zeros(n)
        self._xdot = np.zeros(n)
        self._M_pulled = np.zeros((m, m))
        self._h_pulled = np.zeros(m)
        self._q = np.zeros(m)
        self._qdot = np.zeros(m)

    def update(self, q, qdot):
        self._q = q
        self._qdot = qdot
        self._J = self.J_fun(q)
        self._Jdot = self.J_dot_fun(q, qdot)
        self._Jt = np.transpose(self._J)
        self._x = np.array(self.phi_fun(q))[:, 0]
        self._xdot = np.dot(self._J, qdot)
        self._M = self.M_fun(self._x, self._xdot)
        self._M_pulled = np.dot(self._Jt, np.dot(self._M, self._J))
        h = self._geo.h(self._x, self._xdot)
        h_int = np.dot(self._Jt, np.dot(self._M, (h + np.dot(self._Jdot, self._qdot))))[:, 0]
        self._h_pulled = np.linalg.solve(self._M_pulled, h_int)

    def M_pulled(self):
        return self._M_pulled

    def h_pulled(self):
        return self._h_pulled


class RootGeometry(object):
    def __init__(self, leaves):
        self._leaves = leaves
        self._h = np.zeros(m)
        self._rhs = np.zeros(m)
        self._rhs_aug = np.zeros(2 * m)
        self._q = np.zeros(m)
        self._qdot = np.zeros(m)
        self._M = np.zeros((m, m))

    def update(self, q, qdot):
        self._q = q
        self._qdot = qdot
        self._M = np.zeros((m, m))
        h_int = np.zeros(m)
        for leaf in self._leaves:
            leaf.update(q, qdot)
            self._M += leaf.M_pulled()
            h_int += np.dot(leaf.M_pulled(), leaf.h_pulled())
        self._h = np.linalg.solve(self._M, h_int)

    def setRHS(self):
        self._rhs = -self._h

    def augment(self):
        self._rhs_aug[0] = self._qdot[0]
        self._rhs_aug[1] = self._qdot[1]
        self._rhs_aug[2] = self._rhs[0]
        self._rhs_aug[3] = self._rhs[1]

    def contDynamics(self, z, t):
        self.update(z[0:m], z[m:2 * m])
        self.setRHS()
        self.augment()
        zdot = self._rhs_aug
        return zdot

    def computePath(self, z0, t):
        sol, info = odeint(self.contDynamics, z0, t, full_output=True)
        return sol


def main():
    # setup 
    geo1 = Geometry(h_fun=h1_fun)
    geo2 = Geometry(h_fun=h2_fun)
    leaf1 = Leaf(M1_fun, phi1_fun, J1_fun, J1dot_fun, geo1)
    leaf2 = Leaf(M1_fun, phi1_fun, J1_fun, J1dot_fun, geo2)
    rootGeo1 = RootGeometry([leaf1])
    rootGeo2 = RootGeometry([leaf2])
    rootGeo3 = RootGeometry([leaf1, leaf2])
    w0 = 1.20
    r0_dot = -1.5
    r0 = 2.0
    a0 = 1.0 / 3.0 * np.pi
    q0 = np.array([r0, a0])
    q0_dot = np.array([r0_dot, w0])
    t = np.arange(0.0, 50.00, 0.05)
    z0 = np.concatenate((q0, q0_dot))
    # solving
    sol1 = rootGeo1.computePath(z0, t)
    sol2 = rootGeo2.computePath(z0, t)
    sol3 = rootGeo3.computePath(z0, t)
    # plotting
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("Energization a geometry")
    ax[0][0].set_title("Geometry generator 1")
    ax[0][1].set_title("Geometry generator 2")
    ax[1][0].set_title("Geometry generator 3")
    obst1 = plt.Circle(tuple(x_obst), radius=r_obst, color='k', fill=None, hatch='///')
    obst2 = plt.Circle(tuple(x_obst), radius=r_obst, color='k', fill=None, hatch='///')
    obst3 = plt.Circle(tuple(x_obst), radius=r_obst, color='k', fill=None, hatch='///')
    # ax[0][0].add_patch(obst1)
    ax[0][1].add_patch(obst2)
    ax[1][0].add_patch(obst3)
    (x, y, line, point) = plot_polar_trajectory(sol1, ax[0][0])
    (x2, y2, line2, point2) = plot_polar_trajectory(sol2, ax[0][1])
    (x3, y3, line3, point3) = plot_polar_trajectory(sol3, ax[1][0])
    animation_data = [
        [line, line2, line3],
        [point, point2, point3],
        [{'x': x, 'y': y}, {'x': x2, 'y': y2}, {'x': x3, 'y': y3}]
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
