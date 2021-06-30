import gym
import pointRobot
import time

import matplotlib.pyplot as plt
import numpy as np
import casadi as ca

from optFabrics.controllers.staticController import StaticController

from obstacle import Obstacle
from robotPlot import RobotPlot


def main():
    ## setting up the problem
    n = 2
    q_ca = ca.SX.sym("q", n)
    qdot_ca = ca.SX.sym("qdot", n)
    x0 = np.array([2.0, 2.0])
    xdot_mag = 1.0
    fk = q_ca
    x_d = np.array([-1.0, -1.0])
    # construct fabric controller
    fabCon = StaticController(2, q_ca, qdot_ca)
    fabCon.addAttractor(x_d, 2, fk)
    obsts = [Obstacle(np.array([0.5, 0.8]), 1.0)]
    fabCon.addObstacles(obsts, fk)
    fabCon.addDamper(2, fk)
    fabCon.assembleRootGeometry(m=0.5)
    # setup environment
    con = fabCon
    n_steps = 1000
    qs = []
    alphas = [-np.pi/1 + np.pi/5 * i for i in range(1, 2)]
    ## running the simulation
    for alpha in alphas:
        env = gym.make('point-robot-acc-v0', dt=0.01)
        xdot0 = xdot_mag * np.array([np.cos(alpha), -np.sin(alpha)])
        ob = env.reset(x0, xdot0)
        print("Starting episode")
        q = np.zeros((n_steps, n))
        t = 0.0
        for i in range(n_steps):
            if i % 100 == 0:
                print('time step : ', i)
            t += env._dt
            action = con.computeAction(ob, t)
            #env.render()
            ob, reward, done, info = env.step(action)
            q[i, :] = ob[0:n]
        qs.append(q)
    ## Plotting the results
    fk_fun = lambda q : q
    robotPlot = RobotPlot(qs[0:1], fk_fun, 2, types=[0])
    robotPlot.initFig(1, 1)
    robotPlot.addObstacle([0], obsts)
    robotPlot.plot()
    robotPlot.makeAnimation(n_steps)
    robotPlot.addSolutions(0, qs)
    robotPlot.show()
if __name__ == "__main__":
    main()
