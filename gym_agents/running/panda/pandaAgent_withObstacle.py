import gym
import nLinkUrdfReacher
import time
import os
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
    ## setting up the problem
    limits = getLimits()
    x_d = np.array([0.6, -0.3, 0.5])
    x_d_r = x_d + np.array([0.0, 0.04, 0.00])
    x_d_l = x_d + np.array([0.0, -0.04, 0.00])
    obsts = [Obstacle3D(np.array([0.5, 0.0, 0.80]), 0.15)]
    # construct fabric controller
    q_ca = ca.SX.sym("q", n)
    qdot_ca = ca.SX.sym("qdot", n)
    fk_leftfinger = forwardKinematics(q_ca[0:8], tip="panda_leftfinger")
    fk_rightfinger = forwardKinematics(ca.vertcat(q_ca[0:7], q_ca[8]), tip="panda_rightfinger")
    diff_fk = fk_rightfinger[0:3] - fk_leftfinger[0:3]
    fk_l_fun = ca.Function("fkl", [q_ca[0:8]], [fk_leftfinger])
    fk_r_fun = ca.Function("fkr", [ca.vertcat(q_ca[0:7], q_ca[8])], [fk_rightfinger])
    # motion planner
    con = StaticController(n, q_ca, qdot_ca)
    con.addAttractor(x_d_r[0:3], 3, fk_rightfinger[0:3], k=2.0)
    con.addAttractor(x_d_l[0:3], 3, fk_leftfinger[0:3], k=2.0)
    con.addAttractor(np.array([0.0, 0.08, 0.0]), 3, diff_fk, k=50.0)
    con.addJointLimits(limits[0][:, 0], limits[0][:, 1])
    for i in range(8):
        fk_i = forwardKinematics(q_ca[0:i], tip="panda_link" + str(i))
        con.addObstacles(obsts, fk_i[0:3])
    con.addObstacles(obsts, fk_leftfinger[0:3])
    con.addObstacles(obsts, fk_rightfinger[0:3])
    # Plane is used for collision avoidance with table
    con.addPlane(0.4, fk_leftfinger[2])
    con.addPlane(0.4, fk_rightfinger[2])
    con.addDamper(9, q_ca)
    m = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.01, 0.01])
    con.assembleRootGeometry(m=m)
    n_steps = 1500
    ## running the simulation
    env = gym.make('panda-reacher-acc-v0', dt=0.01, render=True, gripper=True)
    print("Starting episode")
    q0 = np.array([0.8, 0.0, 0.0, -1.501, 0.0, 1.8675, 0.0, 0.02, 0.02])
    t = 0.0
    #ob = env.reset()
    ob = env.reset(q0)
    for obst in obsts:
        posObstacle = [obst.x()[0], obst.x()[1], obst.x()[2] + 0.2]
        env.addObstacle(pos=posObstacle, filename="sphere05red_nocol.urdf")
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
