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
            limitPos[i, 0] = joint_info.limit.lower -0.3
            limitPos[i, 1] = joint_info.limit.upper + 0.3
            limitVel[i, 0] = -joint_info.limit.velocity
            limitVel[i, 1] = joint_info.limit.velocity
            limitTor[i, 0] = -joint_info.limit.effort
            limitTor[i, 1] = joint_info.limit.effort
            i += 1
        if joint_info.type == "prismatic":
            limitPos[i, 0] = joint_info.limit.lower - 1.1
            limitPos[i, 1] = joint_info.limit.upper + 1.1
            limitVel[i, 0] = -joint_info.limit.velocity
            limitVel[i, 1] = joint_info.limit.velocity
            limitTor[i, 0] = -joint_info.limit.effort
            limitTor[i, 1] = joint_info.limit.effort
            i += 1
            limitPos[i, 0] = joint_info.limit.lower - 1.1
            limitPos[i, 1] = joint_info.limit.upper + 1.1
            limitVel[i, 0] = -joint_info.limit.velocity
            limitVel[i, 1] = joint_info.limit.velocity
            limitTor[i, 0] = -joint_info.limit.effort
            limitTor[i, 1] = joint_info.limit.effort
    return limitPos, limitVel, limitTor


def main():
    ## setting up the problem
    limits = getLimits()
    x_d = np.array([0.50, -0.3, 0.40])
    obsts = [Obstacle3D(x_d - np.array([0.0, 0.0, 0.01]), 0.015)]
    finger_off = np.array([0.0, 0.04, 0.0])
    # construct fabric controller
    q_ca = ca.SX.sym("q", n)
    qdot_ca = ca.SX.sym("qdot", n)
    fk_link8 = forwardKinematics(q_ca[0:7], tip="panda_link8")
    fk_leftfinger = forwardKinematics(q_ca[0:8], tip="panda_leftfinger")
    fk_rightfinger = forwardKinematics(ca.vertcat(q_ca[0:7], q_ca[8]), tip="panda_rightfinger")
    dist_finger = ca.norm_2(fk_leftfinger - fk_rightfinger)
    fk_left_xz = ca.vertcat(fk_leftfinger[0], fk_leftfinger[2])
    fk_right_xz = ca.vertcat(fk_rightfinger[0], fk_rightfinger[2])
    fk_l_fun = ca.Function("fkl", [q_ca[0:8]], [fk_leftfinger])
    fk_r_fun = ca.Function("fkr", [ca.vertcat(q_ca[0:7], q_ca[8])], [fk_rightfinger])
    fk_8_fun = ca.Function("fk_8", [q_ca[0:7]], [fk_link8])
    # controller pose1
    con = StaticController(n, q_ca, qdot_ca)
    # con.addAttractor(np.pi/4, 1, q_ca[6], k=5.0)
    con.addAttractor(x_d + finger_off, 3, fk_rightfinger[0:3], k=1.0)
    con.addAttractor(x_d - finger_off, 3, fk_leftfinger[0:3], k=1.0)
    dist_r_fun = ca.Function("dist_r", [ca.vertcat(q_ca[0:7], q_ca[8])], [ca.norm_2(x_d + finger_off - fk_rightfinger)])
    dist_l_fun = ca.Function("dist_l", [q_ca[0:8]], [ca.norm_2(x_d - finger_off - fk_leftfinger)])
    con.addAttractor(0, 1, fk_rightfinger[2] - fk_leftfinger[2], k=100.00)
    con.addAttractor(0, 1, fk_rightfinger[0] - fk_leftfinger[0], k=100.00)
    #con.addAttractor(0.0, 1, fk_link8[1], k=10)
    #con.addAttractor(0.8, 1, q_ca[7] + q_ca[8], k=10)
    con.addAttractor(0.08, 1, dist_finger, k=100)
    con.addJointLimits(limits[0][:, 0], limits[0][:, 1])
    con.addObstacles(obsts, fk_leftfinger)
    con.addObstacles(obsts, fk_rightfinger)
    con.addPlane(0.38, fk_leftfinger[2])
    con.addPlane(0.38, fk_rightfinger[2])
    con.addDamper(9, q_ca)
    con.assembleRootGeometry(m=1.00)
    # controller 2
    con2 = StaticController(n, q_ca, qdot_ca)
    con2.addAttractor(0.0, 1, q_ca[7] + q_ca[8], k=10)
    con2.addPlane(0.38, fk_leftfinger[2])
    con2.addPlane(0.38, fk_rightfinger[2])
    con2.addDamper(9, q_ca)
    con2.assembleRootGeometry(m=1.00)
    n_steps = 3500
    qs = []
    ## running the simulation
    env = gym.make('panda-reacher-acc-v0', dt=0.01, render=True, gripper=True)
    print("Starting episode")
    q = np.zeros((n_steps, n))
    q0 = np.array([0.8, -0.7, 0.0, -1.501, 0.0, 1.8675, np.pi/4, 0.02, 0.02])
    t = 0.0
    #ob = env.reset()
    ob = env.reset(q0)
    env.addObstacle(x_d + np.array([0.0, 0.0, 0.2]), 'cube_small.urdf')
    con2Active = False
    for i in range(n_steps):
        if i % 100 == 0:
            print('time step : ', i)
        t += env._dt
        if con2Active:
            action = con2.computeAction(ob, t)
        else:
            action = con.computeAction(ob, t)
        if np.linalg.norm(action) > 1000:
            print(np.linalg.norm(action))
        #print('action : ', action)
        ob, reward, done, info = env.step(action)
        ee_pos = fk_8_fun(ob[0:7])
        dist2obst = np.linalg.norm(ee_pos - obsts[0].x())
        dr = dist_r_fun(np.concatenate((ob[0:7], ob[8:9])))
        dl = dist_l_fun(np.concatenate((ob[0:7], ob[7:8])))
        #print("dr : ", dr)
        #print("dl : ", dl)
        if (dr + dl) < 0.01:
            print("grasping")
            con2Active = True
        #print("fk_8", ee_pos[2])
        #print("dist : ", dist2obst)
        #print("fkl ", fk_l_fun(ob[0:8]))
        #print("fkr ", fk_r_fun(np.concatenate((ob[0:7], ob[8:9]))))
        #print(ob[6])
        #print("fk : ", fk_fun(ob[0:n]))
        #print("fk1 : ", fk_1_fun(ob[0:n]))
        #print("x_ori : ", x_ori_fun(ob[0:n]))
        q[i, :] = ob[0:n]
    qs.append(q)
    input("wating for end call")
    ## Plotting the results
    """
    fk_fun = lambda q, n : numpyFk(q, n)[0:3]
    robotPlot = RobotPlot(qs, fk_fun, 3, types=[1, 1, 1, 1])
    robotPlot.initFig(2, 2)
    robotPlot.addObstacle([0, 1], obsts)
    robotPlot.plot()
    robotPlot.makeAnimation(n_steps)
    robotPlot.show()
    """
if __name__ == "__main__":
    main()
