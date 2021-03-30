import gym
import nLinkUrdfReacher
import time
import os

import matplotlib.pyplot as plt
import numpy as np
import casadi as ca
import urdf2casadi.urdfparser as u2c


from optFabrics.leaf import Leaf, createAttractor, createCollisionAvoidance
from optFabrics.rootGeometry import RootGeometry
from optFabrics.damper import createRootDamper
from optFabrics.plottingGeometries import plot2DRobot, plotMultiple2DRobot
from optFabrics.diffMap import DiffMap

import pandaReacher

# file names
n = 7
robot = u2c.URDFparser()
urdf_file = os.path.dirname(pandaReacher.__file__) + "/resources/panda.urdf"
robot.from_file(urdf_file)
root = "panda_link0"
tip = "panda_link8"
gravity = np.array([0.0, 0.0, -10.0])
forward_kinematics = robot.get_forward_kinematics(root, tip)


def forwardKinematics(q):
    T_fk = forward_kinematics["T_fk"](q)
    print(T_fk[0:3, 3].size())
    fk = forward_kinematics["T_fk"](q)[0:3, 3]
    return fk

class FabricController():
    def __init__(self, n):
        self._n = n
        m = 3
        q = ca.SX.sym("q", n)
        qdot = ca.SX.sym("qdot", n)
        x = ca.SX.sym("x", m)
        xdot = ca.SX.sym("xdot", m)
        x_d = np.array([0.2, 0.7, 0.4])
        fk = forwardKinematics(q)
        lforcing = createAttractor(q, qdot, x, xdot, x_d, fk, k=5.0)
        x_ex = ca.SX.sym("x_ex", 3)
        xdot_ex = ca.SX.sym("xdot_ex", 3)
        phi_ex = forwardKinematics(q)
        diffMap_ex = DiffMap("exec_map", phi_ex, q, qdot, x_ex, xdot_ex)
        rootDamper = createRootDamper(q, qdot, x, diffMap_ex, x_ex, xdot_ex, b=np.array([5.0, 20.0]))
        le_root = 1.0/2.0 * ca.dot(qdot, qdot)
        self._rg_forced = RootGeometry([lforcing], le_root, n, damper=rootDamper)

    def computeAction(self, z, t):
        q  = z[0:self._n]
        qdot = z[self._n:self._n * 2]
        zdot = self._rg_forced.contDynamics(z, t)
        qddot = zdot[self._n:2 * self._n]
        #print(qddot)
        #print(tau)
        return qddot


def main():
    ## setting up the problem
    con1 = FabricController(n)
    n_steps = 200000
    ## running the simulation
    env = gym.make('panda-reacher-acc-v0', dt=0.01, render=True)
    ob = env.reset()
    print("Starting episode")
    t = 0.0
    for i in range(n_steps):
        if i % 100 == 0:
            print('time step : ', i)
        time.sleep(env._dt)
        t += env._dt
        action = con1.computeAction(ob, t)
        env.render()
        print("fk : ", forwardKinematics(ob[0:n]))
        ob, reward, done, info = env.step(action)

if __name__ == "__main__":
    main()
