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
    xdot0 = np.array([-1.0, -1.0])
    fk = q_ca
    x_d = np.array([-1.0, -1.0])
    # construct fabric controller
    fabCon = StaticController(2, q_ca, qdot_ca)
    fabCon.addAttractor(x_d, 2, fk)
    obsts = [Obstacle(np.array([0.5, 0.8]), 0.3), Obstacle(np.array([1.2, -0.5]), 0.2)]
    fabCon.addObstacles(obsts, fk)
    fabCon.addDamper(2, fk)
    fabCon.assembleRootGeometry()
    # setup environment
    cons = [fabCon]
    n_steps = 1000
    qs = []
    ## running the simulation
    for i in range(len(cons)):
        con = cons[i]
        env = gym.make('point-robot-acc-v0', dt=0.01)
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
    robotPlot = RobotPlot(qs, fk_fun, 2, types=[0])
    robotPlot.initFig(2, 2)
    robotPlot.addObstacle([0], obsts)
    robotPlot.plot()
    robotPlot.makeAnimation(n_steps)
    robotPlot.show()
if __name__ == "__main__":
    main()
