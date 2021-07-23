import gym
import pointRobot
import time


import casadi as ca
import numpy as np
from optFabrics.diffGeometry.fabricPlanner import FabricPlanner
from optFabrics.diffGeometry.diffMap import DifferentialMap
from optFabrics.diffGeometry.energy import FinslerStructure, Lagrangian
from optFabrics.diffGeometry.geometry import Geometry

from obstacle import Obstacle
from robotPlot import RobotPlot


def main():
    ## setting up the problem
    obsts = [
                Obstacle(np.array([0.0, 0.0]), 1.0),
            ]
    n = 2
    q = ca.SX.sym("q", n)
    qdot = ca.SX.sym("qdot", n)
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    l_base = 0.5 * ca.dot(qdot, qdot)
    h_base = ca.SX(np.zeros(n))
    baseGeo = Geometry(h=h_base, x=q, xdot=qdot)
    baseLag = Lagrangian(l_base, q, qdot)
    planner = FabricPlanner(baseGeo, baseLag)
    phi = ca.norm_2(q - obsts[0].x()) / obsts[0].r() - 1
    dm = DifferentialMap(q, qdot, phi)
    s = -0.5 * (ca.sign(xdot) - 1)
    lam = 5.00
    le = lam * 1/x * s * xdot**2
    lag_col = Lagrangian(le, x, xdot)
    h = -lam / (x ** 3) * xdot**2
    geo = Geometry(h=h, x=x, xdot=xdot)
    planner.addGeometry(dm, lag_col, geo)
    l_ex = 0.5 * ca.dot(qdot, qdot)
    exLag = Lagrangian(l_ex, q, qdot)
    exLag.concretize()
    planner.setExecutionEnergy(exLag)
    planner.concretize()
    # setup environment
    cons = [planner]
    n_steps = 100
    qs = []
    x0 = np.array([2.3, 0.5])
    xdot0 = np.array([-1.0, -0.0])
    # running the simulation
    for i in range(len(cons)):
        con = cons[i]
        env = gym.make('point-robot-acc-v0', dt=0.01)
        ob = env.reset(x0, xdot0)
        print("Starting episode")
        q = np.zeros((n_steps, n))
        t = 0.0
        for i in range(n_steps):
            if i % 100 == 0:
                print('time step : ', i)
            t += env._dt
            # t0 = time.time()
            action = con.computeActionEx(ob[0:2], ob[2:4])
            _, _, en_ex = exLag.evaluate(ob[0:2], ob[2:4])
            #print(en_ex)
            # print(time.time() - t0)
            # env.render()
            ob, reward, done, info = env.step(action)
            q[i, :] = ob[0:n]
        qs.append(q)
    ## Plotting the results
    fk_fun = lambda q : q
    robotPlot = RobotPlot(qs, fk_fun, 2, types=[0])
    robotPlot.initFig(2, 2)
    robotPlot.addObstacle([0], obsts)
    robotPlot.plot()
    robotPlot.makeAnimation(n_steps)
    robotPlot.show()

if __name__ == "__main__":
    main()
