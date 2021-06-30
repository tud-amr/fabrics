import gym
import groundRobots
import time

import matplotlib.pyplot as plt
import numpy as np
import casadi as ca
import math

from obstacle import Obstacle
from robotPlot import RobotPlot


from geomdl import BSpline
from geomdl import utilities
from geomdl.visualization import VisMPL

from optFabrics.controllers.dynamicController import DynamicController

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    #return min(np.arccos(np.dot(v1_u, v2_u)), np.arccos(np.dot(v1_u, -v2_u)))
    return np.arctan(v1_u[1]/v1_u[0]) - np.arctan(v2_u[1]/v2_u[0])
    #return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def projectAction(z, zddot_des):
    qddot = zddot_des
    P = np.array([np.cos(z[2]), np.sin(z[2])])
    u = np.dot(P, qddot[0:2])
    #print("theta : ", z[2])
    #print("norm des acc : ", np.linalg.norm(qddot[0:2]))
    print("des acc angle : ", math.atan2(qddot[1], qddot[0]))
    print("possible acc angle : ", z[2])
    angle = -z[2] + math.atan2(qddot[1], qddot[0])
    angle_gain = 10.0
    theta_ddot = qddot[2] + angle * angle_gain
    return np.array([u, qddot[2]])

def main():
    n = 3
    q_ca = ca.SX.sym("q", 3)
    qdot_ca = ca.SX.sym("qdot", 3)
    obsts = [Obstacle(np.array([3.0, -0.2]), 0.35)]
    ## setting up the problem
    x0 = np.array([0.0, 0.0])
    x_d = np.array([5.0, 5.0])
    fk = q_ca[0:3]
    con = DynamicController(3, q_ca, qdot_ca)
    crv = BSpline.Curve()
    crv.degree = 4
    crv.ctrlpts = [x0.tolist(), [3.0, 0.0], [4.5, 0.0], [5.0, 1.0], x_d.tolist()]
    crv.knotvector = utilities.generate_knot_vector(crv.degree, len(crv.ctrlpts))
    """
    vis_config = VisMPL.VisConfig(legend=False, axes=False, figure_dpi=120)
    vis_obj = VisMPL.VisCurve2D(vis_config)
    crv.vis = vis_obj
    crv.render()
    """
    con.addDiffDriveAttractor(crv, 3, fk[0:3], 20)
    con.addObstacles(obsts, fk[0:2])
    #con.addAttractor(np.array([np.pi/4]), 1, q_ca[2], k=5.0)
    #con.addLeaf(geo_leaf)
    con.addDamper(3, q_ca)
    m = np.identity(3)
    con.assembleRootGeometry()
    ## running the simulation
    env = gym.make('ground-robot-diffdrive-acc-v0', dt = 0.01)
    n_steps = 2000
    t = 0.0
    ob = env.reset()
    qs = []
    q = np.zeros((n_steps, n))
    for i in range(n_steps):
        if i % 1000 == 0:
            print('time step : ', i)
        t += env._dt
        #time.sleep(env._dt)
        action_des = con.computeAction(ob, t)
        #print("action_des : ", action_des)
        action_proj = projectAction(ob[0:3], action_des)
        #print("action : ", action_proj)
        env.render()
        ob, reward, done, info = env.step(action_proj)
        q[i, :] = ob[0:n]
    qs.append(q)
    fk_fun = lambda q : q[0:2]
    robotPlot = RobotPlot(qs, fk_fun, 2, types=[0])
    robotPlot.initFig(2, 2)
    robotPlot.addObstacle([0], obsts)
    robotPlot.plot()
    robotPlot.addSpline([0], crv)
    robotPlot.makeAnimation(n_steps)
    robotPlot.show()

if __name__ == "__main__":
    main()
