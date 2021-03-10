import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import casadi as ca

n = 2

z = ca.MX.sym('z', 4)
w = ca.norm_2(z[2:4])/ca.norm_2(z[0:2])
a = ca.arctan2(z[1], z[0])
r = ca.norm_2(z[0:2])
b = 1.0
h0 = -b * r * w**2 * ca.cos(a)
h1 = -b * r* w**2 * ca.sin(a)

h = ca.vertcat(h0, h1)


class Geometry(object):

    """Geometry as in Optimization fabrics
        xddot + h(x, xdot) = 0
    """

    def __init__(self):
        self._z = z

    def setRHS(self):
        self._rhs = h

    def augment(self):
        self._rhs_aug = ca.vertcat(self._z[2], self._z[3], self._rhs[0], self._rhs[1])

    def createSolver(self, dt=0.01):
        self._dt = dt
        self.setRHS()
        self.augment()
        ode = {}
        ode['x'] = self._z
        ode['ode'] = self._rhs_aug
        self._int_fun = ca.integrator("int_fun", 'cvodes', ode, {'tf':dt})

    def computePath(self, z0, T):
        num_steps = int(T/self._dt)
        z = z0
        sols = np.zeros((num_steps, 4))
        for i in range(num_steps):
            res = self._int_fun(x0=z)
            z = np.array(res['xf'])[:, 0]
            sols[i, :] = z
        return sols

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

def main():
    # setup 
    geo = Geometry()
    geo.createSolver(dt=0.001)
    w0 = 1.0
    r0 = 2.0
    a0 = 1.0/3.0 * np.pi
    q0 = r0 * np.array([np.cos(a0), np.sin(a0)])
    q0_dot = np.array([-r0 * w0 * np.sin(a0), r0 * w0 * np.cos(a0)])
    z0 = np.concatenate((q0, q0_dot))
    # solving
    sol = geo.computePath(z0, 10.0)
    # plotting
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle("Energization a geometry")
    ax[0][0].set_title("Original geometry generator")
    ax[0][1].set_title("Energized geometry generator")
    (x, y, line, point) = plotTraj(sol, ax[0][0], fig)
    (x2, y2, line2, point2) = plotTraj(sol, ax[0][1], fig)
    ani = animation.FuncAnimation(
        fig, update, len(x),
        fargs=[x, x2, y, y2, line, line2, point, point2],
        interval=25, blit=True
    )
    plt.show()


if __name__ == "__main__":
    #cProfile.run('main()', 'restats_with')
    main()
