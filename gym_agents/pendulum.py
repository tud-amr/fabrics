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
    q_d = np.array([0.0, 0.0])
    x_d = np.array([1.7, 0.0])
    q_ca = ca.SX.sym("q", 2)
    qdot_ca = ca.SX.sym("qdot", 2)
    con = StaticController(2, q_ca, qdot_ca)
    fk = casadiFk(q_ca, 2)
    con.addAttractor(x_d, 2, fk[0:2])
    obsts = [Obstacle(np.array([1.0, 2.0]), 0.4)]
    con.addObstacles(obsts, fk[0:2])
    con.addDamper(2, q_ca)
    con.assembleRootGeometry(m=0.2)
    ## running the simulation
    n_steps = 1000
    env = gym.make('nLink-reacher-acc-v0', n=2, dt=0.010)
    q0 = np.array([2.0, 0.0])
    qdot0 = np.zeros(2)
    ob = env.reset(pos=q0, vel=qdot0)
    print("Starting episode")
    qs = []
    q = np.zeros((n_steps, 2))
    t = 0.0
    for i in range(n_steps):
        if i % 100 == 0:
            print('time step : ', i)
        t += env._dt
        #env.render()
        action = con.computeAction(ob, t)
        #print('action : ', action)
        ob, reward, done, info = env.step(action)
        #print("q : ", ob[0])
        q[i, :] = ob[0:2]
    qs.append(q)
    ## Plotting the results
    fk_fun = lambda q, n : numpyFk(q, n)
    robotPlot = RobotPlot(qs, fk_fun, 2, types=[1])
    robotPlot.initFig(1, 1)
    robotPlot.addObstacle([0], obsts)
    robotPlot.plot()
    robotPlot.makeAnimation(n_steps)
    robotPlot.show()
if __name__ == "__main__":
    main()
