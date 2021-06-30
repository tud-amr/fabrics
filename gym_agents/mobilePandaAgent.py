import gym
import mobileReacher
import time
import os
import numpy as np
import casadi as ca
import urdf2casadi.urdfparser as u2c

from optFabrics.controllers.staticController import StaticController

from obstacle import Obstacle

# file names
n = 12
robot = u2c.URDFparser()
urdf_file = os.path.dirname(mobileReacher.__file__) + "/resources/mobilePanda.urdf"
urdf_file = os.path.dirname(mobileReacher.__file__) + "/resources/mobilePandaWithGripper.urdf"
robot.from_file(urdf_file)
gravity = np.array([0.0, 0.0, -10.0])


def forwardKinematics(q, tip='panda_link8', root='world'):
    return robot.get_forward_kinematics(root, tip)["T_fk"](q)[0:3, 3]


def getLimits():
    root = "world"
    tip = "panda_link8"
    joint_infos = robot.get_joint_info(root, tip)[0]
    limitPos = np.zeros((n, 2))
    limitVel = np.zeros((n, 2))
    limitTor = np.zeros((n, 2))
    i = 0
    for joint_info in joint_infos:
        if joint_info.type == "revolute" or joint_info.type == "prismatic":
            print(joint_info.name)
            limitPos[i, 0] = joint_info.limit.lower
            limitPos[i, 1] = joint_info.limit.upper
            limitVel[i, 0] = -joint_info.limit.velocity
            limitVel[i, 1] = joint_info.limit.velocity
            limitTor[i, 0] = -joint_info.limit.effort
            limitTor[i, 1] = joint_info.limit.effort
            i += 1
    root = "world"
    tip = "panda_leftfinger"
    joint_infos = robot.get_joint_info(root, tip)[0]
    limitPos[i, 0] = joint_infos[-1].limit.lower -0.1
    limitPos[i, 1] = joint_infos[-1].limit.upper + 0.1
    limitVel[i, 0] = -joint_infos[-1].limit.velocity
    limitVel[i, 1] = joint_infos[-1].limit.velocity
    limitTor[i, 0] = -joint_infos[-1].limit.effort
    limitTor[i, 1] = joint_infos[-1].limit.effort
    i += 1
    root = "world"
    tip = "panda_rightfinger"
    joint_infos = robot.get_joint_info(root, tip)[0]
    limitPos[i, 0] = joint_infos[-1].limit.lower -0.1
    limitPos[i, 1] = joint_infos[-1].limit.upper + 0.1
    limitVel[i, 0] = -joint_infos[-1].limit.velocity
    limitVel[i, 1] = joint_infos[-1].limit.velocity
    limitTor[i, 0] = -joint_infos[-1].limit.effort
    limitTor[i, 1] = joint_infos[-1].limit.effort
    return limitPos, limitVel, limitTor


def main():
    ## setting up the problem
    obsts_ee = [Obstacle(np.array([1.6, 1.4, 0.6]), 0.05),
        Obstacle(np.array([1.6, 1.4, 0.4]), 0.2)]
    obsts_base = [Obstacle(np.array([1.0, 1.0, 0.2]), 0.45)]
    limits = getLimits()
    x_d_r = np.array([1.56, 1.37, 0.8])
    x_d_l = np.array([1.64, 1.37, 0.8])
    # construct fabric controller
    q_ca = ca.SX.sym("q", n)
    qdot_ca = ca.SX.sym("qdot", n)
    fk_leftfinger = forwardKinematics(q_ca[0:11], tip="panda_leftfinger")
    fk_rightfinger = forwardKinematics(ca.vertcat(q_ca[0:10], q_ca[11]), tip="panda_rightfinger")
    fk_l_fun = ca.Function("fkl", [q_ca[0:8]], [fk_leftfinger])
    fk_r_fun = ca.Function("fkr", [ca.vertcat(q_ca[0:7], q_ca[8])], [fk_rightfinger])
    fk_finger_z = fk_leftfinger[2] - fk_rightfinger[2]
    fk_base = forwardKinematics(q_ca[0:3], tip="base_link")
    # construct fabric controller
    con = StaticController(n, q_ca, qdot_ca)
    con.addAttractor(x_d_r[0:3], 3, fk_rightfinger[0:3], k=2.0)
    con.addAttractor(x_d_l[0:3], 3, fk_leftfinger[0:3], k=2.0)
    con.addAttractor(0, 1, fk_finger_z, k=10.0)
    con.addJointLimits(limits[0][:, 0], limits[0][:, 1])
    con.addDamper(12, q_ca)
    con.assembleRootGeometry(m=0.5)
    # controller grasp
    x_d_r = np.array([1.56, 1.37, 0.7])
    x_d_l = np.array([1.64, 1.37, 0.7])
    con2 = StaticController(n, q_ca, qdot_ca)
    con2.addAttractor(x_d_r[0:3], 3, fk_rightfinger[0:3], k=2.0)
    con2.addAttractor(x_d_l[0:3], 3, fk_leftfinger[0:3], k=2.0)
    con2.addJointLimits(limits[0][:, 0], limits[0][:, 1])
    con2.addDamper(12, q_ca)
    con2.assembleRootGeometry(m=0.5)
    # controller grasp
    x_d_r = np.array([1.6, 1.37, 0.7])
    x_d_l = np.array([1.6, 1.37, 0.7])
    con3 = StaticController(n, q_ca, qdot_ca)
    con3.addAttractor(x_d_r[0:3], 3, fk_rightfinger[0:3], k=2.0)
    con3.addAttractor(x_d_l[0:3], 3, fk_leftfinger[0:3], k=2.0)
    con3.addJointLimits(limits[0][:, 0], limits[0][:, 1])
    con3.addDamper(12, q_ca)
    con3.assembleRootGeometry(m=0.5)
    n_steps = 2000
    qs = []
    ## running the simulation
    env = gym.make('mobile-reacher-acc-v0', dt=0.01, render=True, gripper=True)
    print("Starting episode")
    q = np.zeros((n_steps, n))
    q0 = np.array([0.0, 0.0, 0.0, 0.8, 0.7, 0.0, -1.501, 0.0, 1.8675, 0.0, 0.0, 0.0])
    t = 0.0
    #ob = env.reset()
    ob = env.reset(q0)
    env.addObstacle(obsts_base[0].x())
    con.addObstacles(obsts_base, fk_base)
    con2.addObstacles(obsts_base, fk_base)
    con3.addObstacles(obsts_base, fk_base)
    input("press enter to continue")
    for i in range(n_steps):
        if i % 100 == 0:
            print('time step : ', i)
        t += env._dt
        if i < 1000:
            action = con.computeAction(ob, t)
        elif i < 1200:
            action = con2.computeAction(ob, t)
        else:
            action = con3.computeAction(ob, t)
        ob, reward, done, info = env.step(action)
        #print("fk : ", fk_fun(ob[0:n]))
        #print("fk1 : ", fk_1_fun(ob[0:n]))
        #print("x_ori : ", x_ori_fun(ob[0:n]))
        q[i, :] = ob[0:n]
    qs.append(q)
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
