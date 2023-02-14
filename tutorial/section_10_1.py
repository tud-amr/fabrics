import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import casadi as ca

m = 2
# q are the root coordinates q = [x, y]
q = ca.SX.sym("q", m)
qdot = ca.SX.sym("qdot", m)


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
    return M, f


# differential map and jacobian phi : Q -> Q1
q_obst = np.array([0.0, 0.0])
r_obst = 1.0
phi = (ca.norm_2(q - q_obst) - r_obst) / r_obst
phi_fun = ca.Function("phi", [q, qdot], [phi])
k = 0.5
lam = 0.7
psi = k / phi**2

J = ca.jacobian(phi, q)
Jdot = -ca.jacobian(ca.mtimes(J, qdot), q)
J_fun = ca.Function("J", [q], [J])
Jdot_fun = ca.Function("Jdot1", [q, qdot], [Jdot])

# Explicit/hand-crafted geometry
h1_exp = lam * ca.norm_2(qdot)**2 * ca.gradient(psi, q)
h1_exp_fun = ca.Function("h1_exp", [q, qdot], [h1_exp])

# CHOMP geometry
L_chomp = 0.5 * psi * ca.norm_2(qdot)**2
M_chomp, f_chomp = generateLagrangian(L_chomp, "CHOMP")
h1_chomp = ca.mtimes(ca.pinv(M_chomp), -f_chomp)
h1_chomp_fun = ca.Function('h1_chomp', [q, qdot], [h1_chomp])

# Finsler energy
L_fin = 1/(2 * phi**2) * ca.norm_2(ca.mtimes(J, qdot))**2
h1_fin = lam * L_fin * ca.gradient(psi, q)
h1_fin_fun = ca.Function('h1_fin', [q, qdot], [h1_fin])

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



def update(num, x1, y1, line1, point1, x2, y2, line2, point2, x3, y3, line3, point3):
    start = max(0, num - 100)
    line1.set_data(x1[start:num], y1[start:num])
    point1.set_data(x1[num], y1[num])
    line2.set_data(x2[start:num], y2[start:num])
    point2.set_data(x2[num], y2[num])
    line3.set_data(x3[start:num], y3[start:num])
    point3.set_data(x3[num], y3[num])
    return line1, point1, line2, point2, line3, point3

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

    geo_exp = Geometry(h_fun=h1_exp_fun)
    geo_chomp = Geometry(h_fun=h1_chomp_fun)
    geo_fin = Geometry(h_fun=h1_fin_fun)
    geos = [geo_exp, geo_chomp, geo_fin]
    step_size = 0.4
    number = 12
    y0s = [-step_size * number/2 + step_size/2 + step_size * i for i in range(number)]
    q0_dot = np.array([-1.0, 0.0])
    t = np.arange(0.0, 50.00, 0.02)
    # solving
    sols = []
    for geo in geos:
        geoSols = []
        for y0 in y0s:
            q0 = np.array([3.5, y0])
            z0 = np.concatenate((q0, q0_dot))
            geoSols.append(geo.computePath(z0, t))
        sols.append(geoSols)
    sol1 = sols[0][0]
    sol2 = sols[1][0]
    sol3 = sols[2][0]
    # plotting
    fig, ax = plt.subplots(1, 3, figsize=(10, 10))
    fig.suptitle("Different geometries")
    ax[0].set_title("Explicit geometry")
    ax[1].set_title("CHOMP-like geometry")
    ax[2].set_title("Finsler geometry")
    obst1 = plt.Circle(q_obst, radius=r_obst, color='r')
    obst2 = plt.Circle(q_obst, radius=r_obst, color='r')
    obst3 = plt.Circle(q_obst, radius=r_obst, color='r')
    ax[0].add_patch(obst1)
    ax[1].add_patch(obst2)
    ax[2].add_patch(obst3)
    plotMultipleTraj(sols[0], ax[0], fig)
    plotMultipleTraj(sols[1], ax[1], fig)
    plotMultipleTraj(sols[2], ax[2], fig)
    (x, y, line, point) = plotTraj(sol1, ax[0], fig, ani = True)
    (x2, y2, line2, point2) = plotTraj(sol2, ax[1], fig, ani = True)
    (x3, y3, line3, point3) = plotTraj(sol3, ax[2], fig, ani = True)
    ani = animation.FuncAnimation(
        fig, update, len(x),
        fargs=[x, y, line, point, x2, y2, line2, point2, x3, y3, line3, point3],
        interval=10, blit=True
    )
    plt.show()


if __name__ == "__main__":
    #cProfile.run('main()', 'restats_with')
    main()
