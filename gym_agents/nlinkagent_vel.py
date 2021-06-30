import gym
import nLinkReacher
import time

import matplotlib.pyplot as plt
import numpy as np
import casadi as ca

from optFabrics.controllers.staticController import StaticController

from obstacle import Obstacle
from robotPlot import RobotPlot

from numpyFk import numpyFk
from casadiFk import casadiFk

def main():
    ## setting up the problem
    x_d = np.array([2.0, -2.0])
    obsts = [Obstacle(np.array([1.3, 0.6]), 0.3),
                Obstacle(np.array([0.75, -1.0]), 0.2), 
                Obstacle(np.array([-0.5, -0.5]), 0.2)]
    obsts = [Obstacle(np.array([1.25, -1.3]), 0.4)]
    # construct fabric controller
    ns = [3]
    cons = []
    dims = []
    for n in ns:
        lower_lim = -np.pi/2 * np.ones(n)
        upper_lim = np.pi/2 * np.ones(n)
        q_ca = ca.SX.sym("q", n)
        qdot_ca = ca.SX.sym("qdot", n)
        fk = casadiFk(q_ca, n)[0:2]
        con = StaticController(n, q_ca, qdot_ca)
        con.addAttractor(x_d, 2, fk)
        con.addRedundancyRes()
        con.addJointLimits(lower_lim, upper_lim)
        con.addDamper(2, fk)
        for i in range(n+1):
            fk_col = casadiFk(q_ca, i)[0:2]
            con.addObstacles(obsts, fk_col)
        con.assembleRootGeometry()
        cons.append(con)
        dims.append(n)
    for n in ns:
        lower_lim = -np.pi/2 * np.ones(n)
        upper_lim = np.pi/2 * np.ones(n)
        q_ca = ca.SX.sym("q", n)
        qdot_ca = ca.SX.sym("qdot", n)
        fk = casadiFk(q_ca, n)[0:2]
        con = StaticController(n, q_ca, qdot_ca)
        con.addJointLimits(lower_lim, upper_lim)
        con.addAttractor(x_d, 2, fk)
        con.addRedundancyRes()
        con.addDamper(2, fk)
        con.assembleRootGeometry()
        cons.append(con)
        dims.append(n)
    n_steps = 100
    qs = []
    dt = 0.05
    q0 = np.zeros(n)
    q0dot = np.ones(n)
    ## running the simulation
    for i in range(len(cons)):
        con = cons[i]
        dim = dims[i]
        env = gym.make('nLink-reacher-acc-v0', n=dim, dt=dt)
        ob = env.reset(q0, q0dot)
        print("Starting episode")
        q = np.zeros((n_steps, dim))
        t = 0.0
        for i in range(n_steps):
            if i % 100 == 0:
                print('time step : ', i)
            t += env._dt
            action = con.computeAction(ob, t)
            #env.render()
            ob, reward, done, info = env.step(action)
            q[i, :] = ob[0:dim]
        qs.append(q)
    for i in range(len(cons)):
        con = cons[i]
        dim = dims[i]
        env = gym.make('nLink-reacher-vel-v0', n=dim, dt=dt)
        ob = env.reset(q0, q0dot)
        print("Starting episode")
        q = np.zeros((n_steps, dim))
        t = 0.0
        for i in range(n_steps):
            if i % 100 == 0:
                print('time step : ', i)
            t += env._dt
            cur_vel = ob[dim:2*dim]
            action_acc = con.computeAction(ob, t)
            next_vel = cur_vel + dt * action_acc
            #env.render()
            ob, reward, done, info = env.step(next_vel)
            q[i, :] = ob[0:dim]
        qs.append(q)
    ## Plotting the results
    fk_fun = lambda q, n : numpyFk(q, n)[0:3]
    robotPlot = RobotPlot(qs, fk_fun, 3, types=[1, 1, 1, 1])
    robotPlot.initFig(2, 2)
    robotPlot.addObstacle([0, 2], obsts)
    robotPlot.plot()
    robotPlot.makeAnimation(n_steps)
    robotPlot.show()
if __name__ == "__main__":
    main()
