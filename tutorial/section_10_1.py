import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import casadi as ca

n = 1
m = 2
# q are the root coordinates q = [x, y]
q = ca.SX.sym("q", m)
qdot = ca.SX.sym("qdot", m)

# q is the task space variable
q1 = ca.SX.sym("q1", n)
q1dot = ca.SX.sym("q1dot", n)

# differential map and jacobian phi : Q -> Q1
q_obst = np.array([0.0, 0.0])
r_obst = 1.0
phi = (ca.norm_2(q - q_obst) - r_obst) / r_obst
phi_fun = ca.Function("phi", [q, qdot], [phi])
k = 0.5
lam = 0.7
psi = k / q1**2
psi_exp = k / phi**2

J = ca.jacobian(phi, q)
Jdot = -ca.jacobian(ca.mtimes(J, qdot), q)
J_fun = ca.Function("J", [q], [J])
Jdot_fun = ca.Function("Jdot1", [q, qdot], [Jdot])

h1 = lam * ca.norm_2(q1dot) * ca.gradient(psi, q1)
h1_fun = ca.Function("h1", [q1, q1dot], [h1])

h1_exp = lam * ca.norm_2(qdot) * ca.gradient(psi_exp, q)

h1_exp_fun = ca.Function("h1_exp", [q, qdot], [h1_exp])

M1 = np.identity(n)
M1_fun = ca.Function("M1", [q1, q1dot], [M1])


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

def generateFinslerGeometry(L, name, q, qdot):
    dL_dq = ca.gradient(L, q)
    dL_dqdot = ca.gradient(L, qdot)
    d2L_dq2 = ca.jacobian(dL_dq, q)
    d2L_dqdqdot = ca.jacobian(dL_dq, qdot)
    d2L_dqdot2 = ca.jacobian(dL_dqdot, qdot)

    M = d2L_dqdot2
    F = d2L_dqdqdot
    f_e = -dL_dq
    f = ca.mtimes(ca.transpose(F), qdot) + f_e

    h_neg = ca.mtimes(ca.inv(M), f)
    h = -ca.mtimes(ca.inv(M), f)

    h_fun = ca.Function("h_" + name, [q, qdot], [h])
    h_neg_fun = ca.Function("h_neg_" + name, [q, qdot], [h_neg])
    M_fun = ca.Function("M_" + name, [q, qdot], [M])
    return (h_fun, h_neg_fun, M_fun)

Le_exp = 0.5 * psi_exp * ca.norm_2(qdot)**2
h2_exp_fun, _, _ = generateFinslerGeometry(Le_exp, "2", q, qdot)

Le = 0.5 * psi * ca.norm_2(q1dot)**2
_, h2_neg_fun, M2_fun = generateFinslerGeometry(Le, "2", q1, q1dot)


class Geometry(object):

    def __init__(self, h_fun):
        self.h_fun = h_fun
        self._rhs_aug = np.zeros(4)

    def h(self, x, xdot):
        return self.h_fun(x, xdot)

    def setRHS(self):
        self._rhs = -self._h

    def augment(self, x, xdot):
        self._rhs_aug[0] = xdot[0]
        self._rhs_aug[1] = xdot[1]
        self._rhs_aug[2] = self._rhs[0]
        self._rhs_aug[3] = self._rhs[1]

    def contDynamics(self, z, t):
        self._h = self.h_fun(z[0:2], z[2:4])
        self.setRHS()
        self.augment(z[0:2], z[2:4])
        zdot = self._rhs_aug
        return zdot

    def computePath(self, z0, t):
        sol, info = odeint(self.contDynamics, z0, t, full_output=True)
        return sol


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
        self._x = np.array(self.phi_fun(q, qdot))[:, 0]
        self._xdot = np.dot(self._J, qdot)
        self._M = self.M_fun(self._x, self._xdot)
        self._M_pulled = np.dot(self._Jt, np.dot(self._M, self._J))
        h = self._geo.h(self._x, self._xdot)
        h_int = np.dot(self._Jt, np.dot(self._M, (h + np.dot(self._Jdot, self._qdot))))[:, 0]
        #h_int = np.dot(self._Jt, np.dot(self._M, (h - np.dot(self._Jdot, self._qdot))))[:, 0]
        #h_int = np.dot(self._Jt, np.dot(self._M, (h)))[:, 0]
        self._h_pulled = np.dot(np.linalg.pinv(self._M_pulled), h_int)

    def M_pulled(self):
        return self._M_pulled

    def h_pulled(self):
        return self._h_pulled


class RootGeometry(object):
    def __init__(self, leaves):
        self._leaves = leaves
        self._h = np.zeros(m)
        self._rhs = np.zeros(m)
        self._rhs_aug = np.zeros(2*m)
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
        self._h = np.dot(np.linalg.pinv(self._M), h_int)

    def setRHS(self):
        self._rhs = -self._h

    def augment(self):
        self._rhs_aug[0] = self._qdot[0]
        self._rhs_aug[1] = self._qdot[1]
        self._rhs_aug[2] = self._rhs[0]
        self._rhs_aug[3] = self._rhs[1]

    def contDynamics(self, z, t):
        self.update(z[0:m], z[m:2*m])
        self.setRHS()
        self.augment()
        zdot = self._rhs_aug
        return zdot

    def computePath(self, z0, t):
        sol, info = odeint(self.contDynamics, z0, t, full_output=True)
        return sol

def update(num, x1, y1, line1, point1, x2, y2, line2, point2, x3, y3, line3, point3, x4, y4, line4, point4):
    start = max(0, num - 100)
    line1.set_data(x1[start:num], y1[start:num])
    point1.set_data(x1[num], y1[num])
    line2.set_data(x2[start:num], y2[start:num])
    point2.set_data(x2[num], y2[num])
    line3.set_data(x3[start:num], y3[start:num])
    point3.set_data(x3[num], y3[num])
    line4.set_data(x4[start:num], y4[start:num])
    point4.set_data(x4[num], y4[num])
    return line1, point1, line2, point2, line3, point3, line4, point4

def plotMultipleTraj(sols, ax, fig):
    for sol in sols:
        x = sol[:, 0]
        y = sol[:, 1]
        ax.set_xlim([-4, 4])
        ax.set_ylim([-4, 4])
        ax.plot(x, y)

def plotTraj(sol, ax, fig, ani=False):
    x = sol[:, 0]
    y = sol[:, 1]
    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.plot(x, y)
    if ani:
        (line,) = ax.plot(x, y, color="k")
        (point,) = ax.plot(x, y, "r.")
        return (x, y, line, point)

def main():
    # setup 
    geo1 = Geometry(h_fun=h1_fun)
    geo1_exp = Geometry(h_fun=h1_exp_fun)
    geo2 = Geometry(h_fun=h2_neg_fun)
    geo2_exp = Geometry(h_fun=h2_exp_fun)
    leaf1 = Leaf(M1_fun, phi_fun, J_fun, Jdot_fun, geo1)
    leaf2 = Leaf(M2_fun, phi_fun, J_fun, Jdot_fun, geo2)
    rootGeo1 = RootGeometry([leaf1])
    rootGeo2 = RootGeometry([leaf2])
    geos = [rootGeo1, geo1_exp, rootGeo2, geo2_exp]
    geos = [rootGeo1]
    y0s = [-2 + 0.5 * i for i in range(8)]
    q0_dot = np.array([-1.0, 0.0])
    t = np.arange(0.0, 50.00, 0.02)
    # solving
    sols = []
    for geo in geos:
        geoSols = []
        for y0 in y0s:
            if y0 == 0.0:
                continue
            q0 = np.array([3.5, y0])
            z0 = np.concatenate((q0, q0_dot))
            geoSols.append(geo.computePath(z0, t))
        sols.append(geoSols)
    sol1 = sols[0][6]
    #sol2 = sols[1][6]
    #sol3 = sols[2][6]
    #sol4 = sols[3][6]
    # plotting
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("Energization a geometry")
    ax[0][0].set_title("Variant A with pull")
    #ax[0][1].set_title("Variant A with explicit geometry")
    #ax[1][0].set_title("Variant B with pull")
    #ax[1][1].set_title("Variant B with explicit geometry")
    obst1 = plt.Circle(q_obst, radius=r_obst, color='r')
    #obst2 = plt.Circle(q_obst, radius=r_obst, color='r')
    #obst3 = plt.Circle(q_obst, radius=r_obst, color='r')
    #obst4 = plt.Circle(q_obst, radius=r_obst, color='r')
    ax[0][0].add_patch(obst1)
    ##ax[0][1].add_patch(obst2)
    #ax[1][0].add_patch(obst3)
    #ax[1][1].add_patch(obst4)
    plotMultipleTraj(sols[0], ax[0][0], fig)
    #plotMultipleTraj(sols[1], ax[0][1], fig)
    #plotMultipleTraj(sols[2], ax[1][0], fig)
    #plotMultipleTraj(sols[3], ax[1][1], fig)
    #(x, y, line, point) = plotTraj(sol1, ax[0][0], fig, ani = True)
    #(x2, y2, line2, point2) = plotTraj(sol2, ax[0][1], fig, ani = True)
    #(x3, y3, line3, point3) = plotTraj(sol3, ax[1][0], fig, ani = True)
    #(x4, y4, line4, point4) = plotTraj(sol4, ax[1][1], fig, ani = True)
    """
    ani = animation.FuncAnimation(
        fig, update, len(x),
        fargs=[x, y, line, point, x2, y2, line2, point2, x3, y3, line3, point3, x4, y4, line4, point4],
        interval=10, blit=True
    )
    """
    plt.show()


if __name__ == "__main__":
    #cProfile.run('main()', 'restats_with')
    main()
