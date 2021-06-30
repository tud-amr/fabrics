import gym
import groundRobots
import time

import matplotlib.pyplot as plt
import numpy as np
import casadi as ca

from optFabrics.controllers.staticController import StaticController
from optFabrics.diffMap import DiffMap
from optFabrics.creators.repellers import createCollisionAvoidance

from obstacle import Obstacle
from robotPlot import RobotPlot


class DiffDriveController:
    """description"""
    def __init__(self):
        self._q_ca = ca.SX.sym("q", 3)
        self._qdot_ca = ca.SX.sym("qdot", 3)
        self._q_dd_ca = ca.SX.sym("q_dd", 2)
        self._qdot_dd_ca = ca.SX.sym("qdot_dd", 2)
        self._x_ca = ca.SX.sym("x", 1)
        self._xdot_ca = ca.SX.sym("xdot", 1)
        q0 = np.array([4.0, 0.0])
        r = 0.5
        a = [0.4, 0.2, 20.0, 5.0]
        lam = 0.25
        x_h = self._q_ca[0:2] + 0.1 * ca.vertcat(ca.cos(self._q_ca[2]), ca.sin(self._q_ca[2]))
        phi = (ca.norm_2(x_h - q0) - r)/r
        psi = a[0] / (self._x_ca)**2 + a[1] * ca.log(ca.exp(-a[2] * (self._x_ca - a[3])) + 1)
        self._dm = DiffMap("obst", phi, self._q_ca, self._qdot_ca, self._x_ca, self._xdot_ca)
        h = lam * self._xdot_ca**2 * ca.gradient(psi, self._x_ca)
        self._h = ca.Function("h", [self._x_ca, self._xdot_ca], [h])
        x_att = ca.SX.sym("x_att", 2)
        xdot_att = ca.SX.sym("xdot_att", 2)
        q_att = np.array([8.0, -1.0])
        phi_att = ca.fabs(self._q_ca[0:2] + x_h - q_att)
        k = 0.1
        a_psi = 10.0
        a_m = 0.75
        psi_att = k * (ca.norm_2(x_att) + 1/a_psi * ca.log(1 + ca.exp(-2*a_psi * ca.norm_2(x_att))))
        self._dm_att = DiffMap("att", phi_att, self._q_ca, self._qdot_ca, x_att, xdot_att)
        h_att = ca.gradient(psi_att, x_att)
        self._h_att = ca.Function("h", [x_att, xdot_att], [h_att])


    def computeAction(self, ob):
        q = ob[0:3]
        print("q : ", q)
        qdot = ob[3:6]
        x, xdot, J, Jt, Jdot = self._dm.forwardMap(q, qdot, 0.0)
        x_att, xdot_att, J_att, Jt_att, Jdot_att = self._dm_att.forwardMap(q, qdot, 0.0)
        print('x_att : ', x_att)
        h = np.array(self._h(x, xdot))[:, 0]
        h_att = np.array(self._h_att(x_att, xdot_att))[:, 0]
        print("h_att : ", h_att)
        lam = 0.25
        print("x : ", x)
        print("xdot : ", xdot)
        if xdot < 0:
            M = lam/(x[0]**2)
        else:
            M = 0
        M_att = np.identity(2)
        M_pulled = np.dot(Jt, np.dot(M, J)) + np.identity(3)*0.1
        M_att_pulled = np.dot(Jt_att, np.dot(M_att, J_att)) + np.identity(3)*0.0
        h_pulled = np.dot(Jt, h + np.dot(M, np.dot(Jdot, qdot)))
        h_att_pulled = np.dot(Jt_att, h_att + np.dot(M_att, np.dot(Jdot_att, qdot)))
        J_dd = np.array([[np.sin(q[2]), 0], [np.cos(q[2]), 0], [0, 1]])
        print(" M: ", M_att_pulled)
        print("h : ", h_pulled)
        #qddot_point = np.dot(np.linalg.pinv(M_pulled), h_pulled)
        #print('qddot : ', qddot_point)
        qddot = np.dot(np.linalg.pinv(np.dot(M_pulled, J_dd)), h_pulled + h_att_pulled)
        qddot = np.dot(np.linalg.pinv(np.dot(M_att_pulled, J_dd)), h_att_pulled)
        return  qddot


def main():
    con = DiffDriveController()
    n_steps = 10
    qs = []
    q0 = np.array([0.0, 0.1, 0.0 * np.pi/4])
    q0dot = np.array([1.0, 0.0])
    ## running the simulation
    env = gym.make('ground-robot-diffdrive-acc-v0', dt=0.01)
    ob = env.reset(q0, q0dot)
    print("Starting episode")
    t = 0.0
    for i in range(n_steps):
        if i % 100 == 0:
            print('time step : ', i)
        t += env._dt
        action = con.computeAction(ob)
        print(action)
        env.render()
        ob, reward, done, info = env.step(action)
if __name__ == "__main__":
    main()
