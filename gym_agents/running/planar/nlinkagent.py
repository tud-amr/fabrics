import gym
import nLinkReacher
import time

import matplotlib.pyplot as plt
import numpy as np
import casadi as ca

from optFabrics.controllers.staticController import StaticController

# custom packages from robotUtils
from obstacle import Obstacle
from robotPlot import RobotPlot

from numpyFk import numpyFk
from casadiFk import casadiFk

def main():
    ## setting up the problem
    x_d = np.array([0.5, -2.1])
    obsts = [Obstacle(np.array([0.90, -1.0]), 0.3)]
    # construct fabric controllers
    ns = [4]
    cons = []
    dims = []
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
            fk_col = casadiFk(q_ca, i, endlink=0.5)[0:2]
            con.addObstacles(obsts, fk_col)
        con.assembleRootGeometry()
        cons.append(con)
        dims.append(n)
    n_steps = 1000
    qs = []
    ## running the simulation
    for i in range(len(cons)):
        con = cons[i]
        dim = dims[i]
        env = gym.make('nLink-reacher-acc-v0', n=dim, dt=0.01)
        ob = env.reset()
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
    ## Plotting the results
    fk_fun = lambda q, n : numpyFk(q, n)[0:3]
    robotPlot = RobotPlot(qs, fk_fun, 3, types=[1, 1, 1, 1])
    robotPlot.initFig(1, 2)
    robotPlot.addObstacle([1], obsts)
    robotPlot.plot()
    robotPlot.makeAnimation(n_steps)
    robotPlot.show()
if __name__ == "__main__":
    main()
