import gym
import mobileRobot
import time

import matplotlib.pyplot as plt
import numpy as np
import casadi as ca

from optFabrics.leaf import Leaf, createAttractor, createCollisionAvoidance, createTimeVariantAttractor
from optFabrics.rootGeometry import RootGeometry
from optFabrics.damper import createRootDamper
from optFabrics.plottingGeometries import plot2DRobot, plotMultiple2DRobot
from optFabrics.diffMap import DiffMap

def forwardKinematics(q, x_base, n):
    l = 1.0
    fk0 = 0.0
    fk1 = 0.0
    fk2 = 0.0
    if n >= 1:
        fk0 += x_base[0]
        fk1 += x_base[1]
        fk2 += q[1]
    for i in range(1, n):
        fk0 += ca.cos(fk2) * l
        fk1 += ca.sin(fk2) * l
        if i < (q.size(1)-1):
            fk2 += q[i+1]
    fk = ca.vertcat(fk0, fk1, fk2)
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
        x_base = ca.vertcat(q[0], 1.2)
        fk = forwardKinematics(q, x_base, n)[indices]
        self._fk_fun = ca.Function("fk", [q], [fk])
        lcols = []
        for i in range(n):
            fk_col = forwardKinematics(q, x_base, i)
            for j in range(len(x_obsts)):
                print("obstacles : ", j)
                x_obst = x_obsts[j]
                r_obst = r_obsts[j]
                lcols.append(createCollisionAvoidance(q, qdot, fk_col, x_obst, r_obst))
        lforcing = createAttractor(q, qdot, x, xdot, x_d, fk, k=5.0)
        x_ex = ca.SX.sym("x_ex", 2)
        xdot_ex = ca.SX.sym("xdot_ex", 2)
        phi_ex = forwardKinematics(q, x_base, n)[0:2]
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
    n = 3
    x_d = np.array([-4.0, 2.7])
    indices = [0, 1]
    x_obsts = [np.array([0.5, 0.8]), np.array([1.2, -0.5])]
    r_obsts = [0.5, 0.2]
    con1 = FabricController(n, x_d, indices, [], [])
    cons = [con1]
    envs = []
    envs.append(gym.make('mobile-robot-acc-v0', n=n-1, dt = 0.01))
    #env.render()
    dims = [3]
    n_steps = 1500
    qs = []
    ## running the simulation
    for i in range(len(cons)):
        con = cons[i]
        env = envs[i]
        dim = dims[i]
        ob = env.reset()
        print("Starting episode")
        q = np.zeros((dim, n_steps))
        t = 0.0
        for i in range(n_steps):
            if i % 1000 == 0:
                print('time step : ', i)
            t += env._dt
            time.sleep(env._dt)
            action = con.computeAction(ob, t)
            #action = np.zeros(3)
            env.render()
            ob, reward, done, info = env.step(action)
            q[:, i] = ob[0:dim]
        qs.append(q)
if __name__ == "__main__":
    main()
