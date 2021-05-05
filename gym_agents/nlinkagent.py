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
    x_d = np.array([-1.0, -1.0])
    obsts = [Obstacle(np.array([1.3, 0.6]), 0.3),
                Obstacle(np.array([0.75, -1.0]), 0.2), 
                Obstacle(np.array([-0.5, -0.5]), 0.2)]
    obsts = [Obstacle(np.array([1.3, 0.6]), 0.3)]
    # construct fabric controller
    ns = [3, 2]
    cons = []
    dims = []
    for n in ns:
        q_ca = ca.SX.sym("q", n)
        qdot_ca = ca.SX.sym("qdot", n)
        fk = casadiFk(q_ca, n)[0:2]
        con = StaticController(n, q_ca, qdot_ca)
        con.addAttractor(x_d, 2, fk)
        con.addRedundancyRes()
        con.addDamper(n, q_ca)
        for i in range(n+1):
            fk_col = casadiFk(q_ca, i)[0:2]
            con.addObstacles(obsts, fk_col)
        con.assembleRootGeometry()
        cons.append(con)
        dims.append(n)
    for n in ns:
        q_ca = ca.SX.sym("q", n)
        qdot_ca = ca.SX.sym("qdot", n)
        fk = casadiFk(q_ca, n)[0:2]
        con = StaticController(n, q_ca, qdot_ca)
        con.addAttractor(x_d, 2, fk)
        con.addRedundancyRes()
        con.addDamper(n, q_ca)
        con.assembleRootGeometry()
        cons.append(con)
        dims.append(n)
    n_steps = 2000
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
    robotPlot.initFig(2, 2)
    robotPlot.addObstacle([0, 1], obsts)
    robotPlot.plot()
    robotPlot.makeAnimation(n_steps)
    robotPlot.show()
if __name__ == "__main__":
    main()
