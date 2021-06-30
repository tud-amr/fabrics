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
from optFabrics.controllers.staticController import StaticController

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
    n = 10
    q_ca = ca.SX.sym("q", n)
    qdot_ca = ca.SX.sym("qdot", n)
    obsts = [Obstacle(np.array([3.0, -0.2]), 0.35)]
    ## setting up the problem
    x0 = np.array([0.0, 0.0])
    x_d = np.array([5.0, 1.0])
    l = 0.6
    fk = ca.vertcat(q_ca[0] - ca.sin(q_ca[2]) * l/2, q_ca[1] + ca.cos(q_ca[2]) * l/2, q_ca[2])
    fk_con1 = ca.vertcat(q_ca[0] - q_ca[3] - ca.sin(q_ca[2])* l)
    fk_con2 = ca.vertcat(q_ca[1] - q_ca[4] + ca.cos(q_ca[2])* l)
    fk_con3 = ca.vertcat(q_ca[2] - q_ca[5])
    fk_con4 = ca.vertcat(q_ca[7] - q_ca[6] * ca.tan(q_ca[2]))
    fk_con5 = ca.vertcat(q_ca[9] - q_ca[8] * ca.tan(q_ca[5]))
    con = StaticController(n, q_ca, qdot_ca)
    """
    vis_config = VisMPL.VisConfig(legend=False, axes=False, figure_dpi=120)
    vis_obj = VisMPL.VisCurve2D(vis_config)
    crv.vis = vis_obj
    crv.render()
    """
    #con.addAttractor(np.pi*0/4, 1, fk[2])
    con.addAttractor(x_d, 2, fk[0:2])
    con.addAttractor(0, 1, fk_con1, k=100.0)
    con.addAttractor(0, 1, fk_con2, k=100.0)
    con.addAttractor(0, 1, fk_con3, k=100.0)
    con.addAttractor(0, 1, fk_con4, k=1.0)
    con.addAttractor(0, 1, fk_con5, k=1.0)
    #con.addObstacles(obsts, fk[0:2])
    #con.addAttractor(np.array([np.pi/4]), 1, q_ca[2], k=5.0)
    #con.addLeaf(geo_leaf)
    con.addDamper(n, q_ca)
    con.assembleRootGeometry()
    ## running the simulation
    env = gym.make('ground-robot-twowheels-acc-v0', dt = 0.01)
    n_steps = 2
    t = 0.0
    ob = env.reset()
    qs = []
    q = np.zeros((n_steps, n))
    a = np.zeros(6)
    for i in range(n_steps):
        if i % 1000 == 0:
            print('time step : ', i)
        t += env._dt
        #time.sleep(env._dt)
        z_aug = np.concatenate((ob[0:6], np.array([ob[7], ob[8], ob[10], ob[11]])))
        zdot_aug = np.concatenate((ob[6:12], np.array([a[0], a[1], a[3], a[5]])))
        z_com = np.concatenate((z_aug, zdot_aug))
        print(z_com)
        a = con.computeAction(z_com, t)
        #print("action_des : ", action_des)
        env.render()
        ob, reward, done, info = env.step(a[0:6])
        q[i, :] = ob[0:n]
    qs.append(q)

if __name__ == "__main__":
    main()
