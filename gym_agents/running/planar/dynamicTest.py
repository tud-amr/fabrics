import gym
import nLinkReacher
import time

import matplotlib.pyplot as plt
import numpy as np
import casadi as ca

from optFabrics.controllers.dynamicController import DynamicController

from obstacle import Obstacle
from robotPlot import RobotPlot

from numpyFk import numpyFk
from casadiFk import casadiFk

def main():
    ## setting up the problem
    w = 2.0
    t_ca = ca.SX.sym("t")
    x_d = ca.vertcat(1.5 + 0.7 * ca.sin(w * t_ca), 2 * ca.cos(w * t_ca))
    x_d = np.array([-1.0, 0.0])
    # construct fabric controller
    obsts = [Obstacle(np.array([1.5, 2.0]), 0.3)]
    obsts = []
    # construct fabric controller
    n = 3
    q_ca = ca.SX.sym("q", n)
    qdot_ca = ca.SX.sym("qdot", n)
    fk = casadiFk(q_ca, n)[0:2]
    con = DynamicController(n, q_ca, qdot_ca)
    q_d = np.array([0.0, 0.0, 0.0])
    #con.addAttractor(x_d, t_ca, 2, fk, k=25.0)
    w = 0.3
    x_g = ca.vertcat(2.0, 2.0 * ca.sin(w * t_ca))
    #con.addAttractor(q_g, t_ca, 3, q_ca, k=25)
    con.addAttractor(x_g, t_ca, 2, fk, k=25)
    #con.addStaticAttractor(q_d, 3, q_ca, k=25)
    #con.addStaticAttractor(x_d, 2, fk, k=1)
    #con.addDamper(n, q_ca)
    con.addConstantDamper(beta=1.0)
    con.assembleRootGeometry(m=0.1)
    n_steps = 10000
    dt = 0.01
    qs = []
    ## running the simulation
    env = gym.make('nLink-reacher-acc-v0', n=n, dt=dt)
    ob = env.reset()
    print("Starting episode")
    q = np.zeros((n_steps,n))
    t = 0.0
    for i in range(n_steps):
        print("-------------------- %d ----------------" % i)
        if i % 100 == 0:
            print('time step : ', i)
        t += env._dt
        action = con.computeAction(ob, t)
        print('state : ', ob[0:n])
        print('action : ', action)
        #env.render()
        ob, reward, done, info = env.step(action)
        q[i, :] = ob[0:n]
    qs.append(q)
    ## Plotting the results
    if n_steps > 500:
        fk_fun = lambda q, n : numpyFk(q, n)[0:3]
        robotPlot = RobotPlot(qs, fk_fun, 3, types=[1, 1, 1, 1], dt=env._dt)
        robotPlot.initFig(2, 2)
        x_goal = ca.Function("x_goal", [t_ca], [x_d])
        #robotPlot.addGoal([0, 1, 2, 3], x_goal)
        robotPlot.addObstacle([2, 3], obsts)
        robotPlot.plot()
        robotPlot.makeAnimation(n_steps)
        robotPlot.show()
if __name__ == "__main__":
    main()
