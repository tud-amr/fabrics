import gym
import mobileRobot
import time

import matplotlib.pyplot as plt
import numpy as np
import casadi as ca

from optFabrics.leaf import Leaf, createAttractor, createCollisionAvoidance, createTimeVariantAttractor, GeometryLeaf, createJointLimits
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
    def __init__(self, n, x_d, indices, x_obsts, r_obsts, upwards=False):
        print("Constructor Controller with ", r_obsts)
        self._n = n
        self._x_d = x_d
        q = ca.SX.sym("q", n)
        qdot = ca.SX.sym("qdot", n)
        m = len(indices)
        x = ca.SX.sym("x", m)
        xdot = ca.SX.sym("xdot", m)
        x_base = ca.vertcat(q[0], 1.2)
        ls = []
        # colision
        fk = forwardKinematics(q, x_base, n)
        lcols = []
        for i in range(n):
            fk_col = forwardKinematics(q, x_base, i)
            for j in range(len(x_obsts)):
                print("obstacles : ", j)
                x_obst = x_obsts[j]
                r_obst = r_obsts[j]
                lcols.append(createCollisionAvoidance(q, qdot, fk_col, x_obst, r_obst))
        ls += lcols
        # forcing
        lforcing = createAttractor(q, qdot, x, xdot, x_d, fk[indices], k=5.0)
        self._fk_fun = ca.Function("fk", [q], [fk[indices]])
        ls.append(lforcing)
        # limit avoidance
        lim_up = np.array([5, (1 + 1/10) * np.pi, np.pi/2.0, np.pi/2.0, np.pi/2.0])
        lim_low = np.array([-5, -np.pi/10, -np.pi/2.0, -np.pi/2.0, -np.pi/2.0])
        llimits = createJointLimits(q, qdot, lim_up, lim_low)
        ls += llimits
        # execution energy
        x_ex = ca.SX.sym("x_ex", 2)
        xdot_ex = ca.SX.sym("xdot_ex", 2)
        phi_ex = forwardKinematics(q, x_base, n)[0:2]
        diffMap_ex = DiffMap("exec_map", phi_ex, q, qdot, x_ex, xdot_ex)
        # facing upwards
        if upwards:
            rot_des = -np.pi/2.0
            phi_rot = rot_des - fk[2]
            x_rot = ca.SX.sym("xrot", 1)
            xdot_rot = ca.SX.sym("xdotrot", 1)
            dm = DiffMap("redundancy", phi_rot, q, qdot, x_rot, xdot_rot)
            hr = ca.norm_2(xdot_rot)**2 * x_rot * 100.0
            lamr = 100.0
            ler = ca.norm_2(xdot_rot)**2 * lamr
            lredundancy = GeometryLeaf("redundancy", dm, ler, hr)
            ls.append(lredundancy)
        rootDamper = createRootDamper(q, qdot, x, diffMap_ex, x_ex, xdot_ex)
        le_root = 1.0/2.0 * ca.dot(qdot, qdot)
        self._rg_forced = RootGeometry(ls, le_root, n, damper=rootDamper)
        #self._rg_forced = RootGeometry(ls, le_root, n)

    def computeAction(self, z, t):
        zdot = self._rg_forced.contDynamics(z, t)
        u = zdot[self._n:2 * self._n]
        return u

    def done(self, z):
        q = z[0: self._n]
        qdot = z[self._n: self._n * 2]
        fk = self._fk_fun(q)
        d = np.linalg.norm(self._fk_fun(q) - self._x_d)
        vel = np.linalg.norm(qdot)
        if d < 0.01 and vel < 0.01:
            return True
        return False


def main():
    ## setting up the problem
    n = 5
    x_d_pre = -np.pi/2.0
    indices = [2]
    con_pre = FabricController(n, x_d_pre, indices, [], [])
    x_d = np.array([4.0, 1.3])
    indices = [0, 1]
    con_motion = FabricController(n, x_d, indices, [], [], upwards=True)

    #env.render()
    dim = 3
    n_steps = 1500
    ## running the simulation
    env = gym.make('mobile-robot-acc-v0', n=n-1, dt = 0.01)
    ob = env.reset()
    print("Starting episode")
    t = 0.0
    ## preset
    final = False
    con = con_pre
    for i in range(n_steps):
        if i % 1000 == 0:
            print('time step : ', i)
        t += env._dt
        time.sleep(env._dt)
        action = con.computeAction(ob, t)
        env.render()
        ob, reward, done, info = env.step(action)
        if con.done(ob[0: n*2]):
            if final==True:
                continue
            final = True
            con = con_motion
            print("finished presetting")
if __name__ == "__main__":
    main()
