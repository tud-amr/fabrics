import gym
import pointRobot
import time


import casadi as ca
import numpy as np
from optFabrics.diffGeometry.fabricPlanner import FabricPlanner
from optFabrics.diffGeometry.diffMap import DifferentialMap
from optFabrics.diffGeometry.energy import FinslerStructure, Lagrangian
from optFabrics.diffGeometry.geometry import Geometry
from optFabrics.diffGeometry.speedControl import Damper

from obstacle import Obstacle
from robotPlot import RobotPlot
from solverPlot import SolverPlot


def main():
    ## setting up the problem
    obsts = [
                Obstacle(np.array([0.0, 0.0]), 1.0),
            ]
    n = 2
    q = ca.SX.sym("q", n)
    qdot = ca.SX.sym("qdot", n)
    # base geometry
    l_base = 0.5 * ca.dot(qdot, qdot)
    h_base = ca.SX(np.zeros(n))
    baseGeo = Geometry(h=h_base, x=q, xdot=qdot)
    baseLag = Lagrangian(l_base, q, qdot)
    planner = FabricPlanner(baseGeo, baseLag)
    # collision avoidance
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    phi = ca.norm_2(q - obsts[0].x()) / obsts[0].r() - 1
    dm = DifferentialMap(q, qdot, phi)
    s = -0.5 * (ca.sign(xdot) - 1)
    lam = 2.00
    le = lam * 1/(x**2) * s * xdot**2
    lag_col = Lagrangian(le, x, xdot)
    h = -lam / (x ** 1) * xdot**2
    geo = Geometry(h=h, x=x, xdot=xdot)
    planner.addGeometry(dm, lag_col, geo)
    # forcing term
    x_psi = ca.SX.sym("x_psi", 2)
    xdot_psi = ca.SX.sym("xdot_psi", 2)
    q_d = np.array([-2.0, -1.0])
    phi_psi = q - q_d
    dm_psi = DifferentialMap(q, qdot, phi_psi)
    k_psi = 5
    m = [0.3, 2]
    a_psi = 10
    a_m = 0.75
    M_psi = ((m[1] - m[0]) * ca.exp(-1*(a_m * ca.norm_2(x_psi))**2) + m[0]) * ca.SX(np.identity(2))
    psi = k_psi * (ca.norm_2(x_psi) + 1/a_psi * ca.log(1 + ca.exp(-2 * a_psi * ca.norm_2(x))))
    le_psi = ca.dot(xdot_psi, ca.mtimes(M_psi, xdot_psi))
    lag_psi = Lagrangian(le_psi, x_psi, xdot_psi)
    h_psi = ca.gradient(psi, x_psi)
    geo_psi = Geometry(h=h_psi, x=x_psi, xdot=xdot_psi)
    planner.addForcingGeometry(dm_psi, lag_psi, geo_psi)
    # execution energy
    l_ex = 0.5 * ca.dot(qdot, qdot)
    exLag = Lagrangian(l_ex, q, qdot)
    exLag.concretize()
    planner.setExecutionEnergy(exLag)
    # Speed control
    a_b = 0.5
    r_b = 1.5
    s_beta = 0.5 * (ca.tanh(-a_b * (ca.norm_2(x_psi) - r_b)) + 1)
    a_ex = ca.SX.sym('a_ex', 1)
    a_le = ca.SX.sym('a_le', 1)
    b = [0.01, 6.5]
    beta_fun = s_beta * b[1] + b[0] + ca.fmax(0, a_ex - a_le)
    beta = Damper(beta_fun, a_ex, a_le, x_psi, dm_psi)
    l_ex_d = 2 * l_ex
    a_eta = 0.5
    a_shift = 0.5
    eta = 0.5 * (ca.tanh(-a_eta * (l_ex - l_ex_d) - a_shift) + 1)
    planner.setSpeedControl(beta, eta)
    planner.concretize()
    # setup environment
    n_steps = 5500
    qs = []
    solverTimes = []
    x0s = [np.array([2.3, -1.0 + i * 0.2]) for i in range(11)]
    xdot0s = [np.array([-1.0, -0.0])]
    # running the simulation
    for xdot0 in xdot0s:
        for x0 in x0s:
            env = gym.make('point-robot-acc-v0', dt=0.01)
            ob = env.reset(x0, xdot0)
            print("Starting episode")
            q = np.zeros((n_steps, n))
            t = 0.0
            solverTime = np.zeros(n_steps)
            for i in range(n_steps):
                """
                if i % 100 == 0:
                    print('time step : ', i)
                """
                t += env._dt
                t0 = time.time()
                action = planner.computeAction(ob[0:2], ob[2:4])
                #_, _, en_ex = exLag.evaluate(ob[0:2], ob[2:4])
                #print(en_ex)
                solverTime[i] = time.time() - t0
                # env.render()
                ob, reward, done, info = env.step(action)
                q[i, :] = ob[0:n]
            qs.append(q)
            solverTimes.append(solverTime)
    ## Plotting the results
    fk_fun = lambda q : q
    sol_indices = [0, 9]
    robotPlot = RobotPlot([qs[i] for i in sol_indices], fk_fun, 2, types=[0, 0])
    robotPlot.initFig(2, 2)
    robotPlot.addObstacle([0, 1], obsts)
    robotPlot.plot()
    robotPlot.makeAnimation(n_steps)
    robotPlot.addSolutions(0, qs)
    solverAxs = robotPlot.getAxs([2, 3])
    solverPlot = SolverPlot([solverTimes[i] for i in sol_indices], 1, 1, axs=solverAxs)
    solverPlot.plot()
    robotPlot.show()

if __name__ == "__main__":
    main()
