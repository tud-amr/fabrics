import gym
import nLinkReacher
import time

import matplotlib.pyplot as plt
import numpy as np
import casadi as ca

from optFabrics.controllers.staticController import StaticController

# custom packages from robotUtils
from obstacle import Obstacle

from numpyFk import numpyFk
from casadiFk import casadiFk

def main():
    ## setting up the problem
    n = 3
    x_d = np.array([0.5, -2.1])
    obsts = [Obstacle(np.array([0.90, -1.0]), 0.3)]
    # construct fabric controllers
    lower_lim = -np.pi/2 * np.ones(n)
    upper_lim = np.pi/2 * np.ones(n)
    q_ca = ca.SX.sym("q", n)
    qdot_ca = ca.SX.sym("qdot", n)
    fk = casadiFk(q_ca, n)[0:2]
    con = StaticController(n, q_ca, qdot_ca)
    con.addJointLimits(lower_lim, upper_lim)
    con.addAttractor(x_d, 2, fk)
    con.addDamper(2, fk)
    con.assembleRootGeometry()
    n_steps = 10
    ## running the simulation
    env = gym.make('nLink-reacher-acc-v0', n=n, dt=0.01)
    ob = env.reset()
    for i in range(n_steps):
        action = con.computeAction(ob, 0.0)
        ob, reward, done, info = env.step(action)

if __name__ == "__main__":
    main()
