import gym
import nLinkReacher
import time

import matplotlib.pyplot as plt
import numpy as np
import casadi as ca

from optFabrics.leaf import Leaf, createAttractor, createCollisionAvoidance
from optFabrics.rootGeometry import RootGeometry
from optFabrics.damper import createRootDamper
from optFabrics.plottingGeometries import plot2DRobot, plotMultiple2DRobot
from optFabrics.diffMap import DiffMap

def forwardKinematics(q, n):
    l = np.array([1.0, 1.0, 1.0])
    fkx = 0.0
    fky = 0.2
    for i in range(n):
        angle = 0.0
        for j in range(i+1):
            angle += q[j]
        fkx += ca.cos(angle) * l[i]
        fky += ca.sin(angle) * l[i]
    fk = ca.vertcat(fkx, fky)
    return fk

class FabricController():
    def __init__(self, n, x_d, indices, x_obsts, r_obsts):
        print("Constructor Controller with ", r_obsts)
        self._n = n
        q = ca.SX.sym("q", n)
        qdot = ca.SX.sym("qdot", n)
        m = len(indices)
        x = ca.SX.sym("x", m)
        xdot = ca.SX.sym("xdot", m)
        fk = forwardKinematics(q, n)[indices]
        lcols = []
        for i in range(n):
            fk_col = forwardKinematics(q, i+1)
            for j in range(len(x_obsts)):
                x_obst = x_obsts[j]
                r_obst = r_obsts[j]
                lcols.append(createCollisionAvoidance(q, qdot, fk_col, x_obst, r_obst))
        lforcing = createAttractor(q, qdot, x, xdot, x_d, fk, k=5.0)
        x_ex = ca.SX.sym("x_ex", 2)
        xdot_ex = ca.SX.sym("xdot_ex", 2)
        phi_ex = forwardKinematics(q, n)[0:2]
        diffMap_ex = DiffMap("exec_map", phi_ex, q, qdot, x_ex, xdot_ex)
        rootDamper = createRootDamper(q, qdot, x, diffMap_ex, x_ex, xdot_ex)
        le_root = 1.0/2.0 * ca.dot(qdot, qdot)
        self._rg_forced = RootGeometry([lforcing] + lcols, le_root, n, damper=rootDamper)

    def computeAction(self, z, t):
        zdot = self._rg_forced.contDynamics(z, t)
        u = zdot[self._n:2 * self._n]
        return u


def main():
    ## setting up the problem
    x_d = np.array([-1.0, -1.0])
    indices = [0, 1]
    x_obsts = [np.array([0.5, 0.8]), np.array([1.2, -0.5])]
    r_obsts = [0.3, 0.2]
    con1 = FabricController(3, x_d, indices, [], [])
    con2 = FabricController(3, x_d, indices, x_obsts, r_obsts)
    con3 = FabricController(2, x_d, indices, [], [])
    con4 = FabricController(2, x_d, indices, x_obsts, r_obsts)
    cons = [con1, con2, con3, con4]
    dims = [3, 3, 2, 2]
    n_steps = 1000
    qs = []
    ## running the simulation
    for i in range(len(cons)):
        con = cons[i]
        dim = dims[i]
        env = gym.make('nLink-reacher-acc-v0', n=dim, dt=0.01)
        ob = env.reset()
        print("Starting episode")
        q = np.zeros((dim, n_steps))
        t = 0.0
        for i in range(n_steps):
            if i % 100 == 0:
                print('time step : ', i)
            t += env._dt
            action = con.computeAction(ob, t)
            #env.render()
            ob, reward, done, info = env.step(action)
            q[:, i] = ob[0:dim]
        qs.append(q)
    ## Plotting the results
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0][0].set_xlim([-3, 3])
    ax[0][0].set_ylim([-3, 3])
    ax[0][1].set_xlim([-3, 3])
    ax[0][1].set_ylim([-3, 3])
    ax[1][0].set_xlim([-3, 3])
    ax[1][0].set_ylim([-3, 3])
    ax[1][1].set_xlim([-3, 3])
    ax[1][1].set_ylim([-3, 3])
    obsts = []
    for j in range(len(x_obsts)):
        x_obst = x_obsts[j]
        r_obst = r_obsts[j]
        obsts.append(plt.Circle(x_obst, radius=r_obst, color='r'))
        ax[0][1].add_patch(obsts[j])
    obsts = []
    for j in range(len(x_obsts)):
        x_obst = x_obsts[j]
        r_obst = r_obsts[j]
        obsts.append(plt.Circle(x_obst, radius=r_obst, color='r'))
        ax[1][1].add_patch(obsts[j])
    """
    if len(indices) == 2:
        goal = plt.Circle(x_d, radius=0.02, color='g')
        ax[0][0].add_patch(goal)
    elif len(indices) == 1:
        if indices[0] == 0:
            x = [x_d, x_d]
            y = [-5, 5]
            goal = ax[0][0].plot(x, y, color='g')
        elif indices[0] == 1:
            y = [x_d, x_d]
            x = [-5, 5]
            goal = ax[0][0].plot(x, y, color='g')
    """
    plotMultiple2DRobot(qs, fig, ax, dims)
if __name__ == "__main__":
    main()
