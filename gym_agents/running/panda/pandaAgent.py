import gym
import nLinkUrdfReacher
import time
import os
import sys
import numpy as np
import casadi as ca
import urdf2casadi.urdfparser as u2c

from optFabrics.controllers.staticController import StaticController

import pandaReacher
from obstacle import Obstacle, Obstacle3D

# file names
n = 9
robot = u2c.URDFparser()
urdf_file = os.path.dirname(pandaReacher.__file__) + "/resources/pandaWithGripper.urdf"
robot.from_file(urdf_file)
gravity = np.array([0.0, 0.0, -10.0])


def forwardKinematics(q, tip='panda_link8', root='panda_link0'):
    return robot.get_forward_kinematics(root, tip)["T_fk"](q)[0:3, 3]

def getLimits():
    root = "panda_link0"
    tip = "panda_rightfinger"
    joint_infos = robot.get_joint_info(root, tip)[0]
    limitPos = np.zeros((n, 2))
    limitVel = np.zeros((n, 2))
    limitTor = np.zeros((n, 2))
    i = 0
    for joint_info in joint_infos:
        if joint_info.type == "revolute":
            limitPos[i, 0] = joint_info.limit.lower
            limitPos[i, 1] = joint_info.limit.upper
            limitVel[i, 0] = -joint_info.limit.velocity
            limitVel[i, 1] = joint_info.limit.velocity
            limitTor[i, 0] = -joint_info.limit.effort
            limitTor[i, 1] = joint_info.limit.effort
            i += 1
        if joint_info.type == "prismatic":
            limitPos[i, 0] = joint_info.limit.lower - 0.1
            limitPos[i, 1] = joint_info.limit.upper + 0.1
            limitVel[i, 0] = -joint_info.limit.velocity
            limitVel[i, 1] = joint_info.limit.velocity
            limitTor[i, 0] = -joint_info.limit.effort
            limitTor[i, 1] = joint_info.limit.effort
            i += 1
            limitPos[i, 0] = joint_info.limit.lower - 0.1
            limitPos[i, 1] = joint_info.limit.upper + 0.1
            limitVel[i, 0] = -joint_info.limit.velocity
            limitVel[i, 1] = joint_info.limit.velocity
            limitTor[i, 0] = -joint_info.limit.effort
            limitTor[i, 1] = joint_info.limit.effort
    return limitPos, limitVel, limitTor


def main():
    print(sys.argv)
    if len(sys.argv) == 1:
        n_steps = 1500
        render = True
    else:
        n_steps = int(sys.argv[1])
        render = False
    ## setting up the problem
    limits = getLimits()
    x_d = np.array([0.7, -0.1, 0.5])
    x_d_r = x_d + np.array([0.0, 0.04, 0.00])
    x_d_l = x_d + np.array([0.0, -0.04, 0.00])
    obsts = [Obstacle3D(np.array([0.5, 0.0, 0.58]), 0.01)]
    # construct fabric controller
    q_ca = ca.SX.sym("q", n)
    qdot_ca = ca.SX.sym("qdot", n)
    fk_leftfinger = forwardKinematics(q_ca[0:8], tip="panda_leftfinger")
    fk_rightfinger = forwardKinematics(ca.vertcat(q_ca[0:7], q_ca[8]), tip="panda_rightfinger")
    diff_x = fk_leftfinger[0] - fk_rightfinger[0]
    diff_y = fk_rightfinger[1] - fk_leftfinger[1]
    fk_l_fun = ca.Function("fkl", [q_ca[0:8]], [fk_leftfinger])
    fk_r_fun = ca.Function("fkr", [ca.vertcat(q_ca[0:7], q_ca[8])], [fk_rightfinger])
    # motion planner
    con = StaticController(n, q_ca, qdot_ca)
    con.addAttractor(x_d_r[0:3], 3, fk_rightfinger[0:3], k=2.0)
    con.addAttractor(x_d_l[0:3], 3, fk_leftfinger[0:3], k=2.0)
    #con.addAttractor(0.0, 1, diff_x, k=20.0)
    con.addAttractor(0.08, 1, diff_y, k=20.0)
    #con.addAttractor(0.75, 1, q_ca[6], k=5.0)
    con.addJointLimits(limits[0][:, 0], limits[0][:, 1])
    con.addDamper(9, q_ca)
    m = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.01, 0.01])
    con.assembleRootGeometry(m=m)
    ## running the simulation
    env = gym.make('panda-reacher-acc-v0', dt=0.01, render=render, gripper=True)
    print("Starting episode")
    q0 = np.array([0.8, 0.7, 0.0, -1.501, 0.0, 1.8675, 0.0, 0.02, 0.02])
    t = 0.0
    #ob = env.reset()
    ob = env.reset()
    for i in range(n_steps):
        if i % 100 == 0:
            print('time step : ', i)
            print("fkl ", fk_l_fun(ob[0:8]))
            print("fkr ", fk_r_fun(np.concatenate((ob[0:7], ob[8:9]))))
        t += env._dt
        action = con.computeAction(ob, t)
        ob, reward, done, info = env.step(action)
if __name__ == "__main__":
    main()
