import gym
import groundRobots
import time
import casadi as ca
import numpy as np

from optFabrics.planner.nonHolonomicPlanner import NonHolonomicPlanner
from optFabrics.diffGeometry.diffMap import DifferentialMap
from optFabrics.diffGeometry.energy import Lagrangian
from optFabrics.diffGeometry.geometry import Geometry
from optFabrics.planner.default_geometries import GoalGeometry
from optFabrics.planner.default_energies import GoalLagrangian
from optFabrics.planner.default_energies import ExecutionLagrangian

from obstacle import Obstacle
from robotPlot import RobotPlot
from solverPlot import SolverPlot


def pointMass(n_steps=5000):
    ## setting up the problem
    obsts = [
                Obstacle(np.array([-0.0, 0.4]), 1.0),
                Obstacle(np.array([0.0, -2.4]), 0.4),
                Obstacle(np.array([4.0, -1.4]), 1.0),
                Obstacle(np.array([-3.0, 2.4]), 1.0),
            ]
    n = 3
    x = ca.SX.sym("x", 3)
    xdot = ca.SX.sym("xdot", 3)
    # actions forward velocity
    qdot = ca.SX.sym("qdot", 2)
    x_col = ca.SX.sym("x", 1)
    xdot_col = ca.SX.sym("xdot", 1)
    M_base = np.identity(3) * 1.0
    l = 0.5 * ca.dot(xdot[0:3], ca.mtimes(M_base, xdot[0:3]))
    l_base = Lagrangian(l, x=x, xdot=xdot)
    h_base = ca.SX(np.zeros(3))
    geo_base = Geometry(h=h_base, var=[x, xdot])
    l_front = 0.1
    J_nh = ca.SX(
        np.array([
            [ca.cos(x[2]), -l_front * ca.sin(x[2])],
            [ca.sin(x[2]), +l_front * ca.cos(x[2])],
            [0, 1]
        ])
    )
    f_extra = qdot[0] * qdot[1] * ca.vertcat(-ca.sin(x[2]), ca.cos(x[2]), 0)
    planner = NonHolonomicPlanner(geo_base, l_base, J_nh, qdot, f_extra)
    # collision avoidance
    l_front = 0.800
    l_back = -0.300
    l_side = 0.35
    l_i = 0.1
    x_i = x[0:2] + ca.vertcat(l_i * ca.cos(x[2]), l_i * ca.sin(x[2]))
    x_f = x[0:2] + ca.vertcat(l_front * ca.cos(x[2]), l_front * ca.sin(x[2]))
    x_b = x[0:2] + ca.vertcat(l_back * ca.cos(x[2]), l_back * ca.sin(x[2]))
    x_r = x_i + ca.vertcat(l_side * ca.sin(x[2]), -l_side * ca.cos(x[2]))
    x_l = x_i + ca.vertcat(l_side * -ca.sin(x[2]), l_side * ca.cos(x[2]))
    x_ee = ca.vertcat(x_f, x[2])
    x_ee_fun = ca.Function('x_ee', [x], [x_ee])
    s = -0.5 * (ca.sign(xdot_col) - 1)
    lam = 5.00
    h = -1 / (x_col ** 3) * xdot_col**2
    geo = Geometry(h=h, x=x_col, xdot=xdot_col)
    x_corners = [x_f, x_b, x_r, x_l]
    lams = [1.0, 0.1, 0.1, 0.1]
    for obst in obsts:
        for i, x_corner in enumerate(x_corners):
            phi = ca.norm_2(x_corner - obst.x()) / obst.r() - 1
            dm = DifferentialMap(phi, q=x, qdot=xdot)
            le = lams[i] * 1/(x_col**1) * s * xdot_col**2
            lag_col = Lagrangian(le, x=x_col, xdot=xdot_col)
            planner.addGeometry(dm, lag_col, geo)
    # forcing
    x_d = np.array([4.0, 3.0])
    x_psi = ca.SX.sym("x_psi", 2)
    xdot_psi = ca.SX.sym("xdot_psi", 2)
    phi_psi = x_f - x_d
    dm_psi = DifferentialMap(phi_psi, q=x, qdot=xdot)
    dm_psi.concretize()
    lag_psi = GoalLagrangian(x_psi, xdot_psi, k_psi=25)
    geo_psi = GoalGeometry(x_psi, xdot_psi)
    geo_psi.concretize()
    planner.addForcingGeometry(dm_psi, lag_psi, geo_psi)
    # finalize
    exLag = ExecutionLagrangian(x, xdot)
    exLag.concretize()
    planner.setExecutionEnergy(exLag)
    # Speed control
    ex_factor = 1.0
    # planner.setConstantSpeedControl(beta=0.1)
    planner.setDefaultSpeedControl(x_psi, dm_psi, exLag, ex_factor)
    planner.concretize()
    # setup environment
    qs = []
    solverTimes = []
    # running the simulation
    env = gym.make('ground-robot-diffdrive-acc-v0', dt=0.005)
    ob = env.reset(np.array([-5.0, 1.0, 0.0]), np.array([1.0, 0.0]))
    print("Starting episode")
    q = np.zeros((n_steps, n))
    t = 0.0
    solverTime = np.zeros(n_steps)
    for i in range(n_steps):
        if i % 1000 == 0:
            print('time step : ', i)
        t += env._dt
        t0 = time.time()
        x = ob[0:3]
        xdot = ob[3:6]
        qdot = ob[6:8]
        action = planner.computeAction(x, xdot, qdot)
        #action = np.array([-0.1, 0.1])
        x_psi, J_psi, Jdot_psi = dm_psi.forward(x, xdot)
        xdot_psi = np.dot(J_psi, xdot)
        h_psi, _ = geo_psi.evaluate(x_psi, xdot_psi)
        #_, _, en_ex = exLag.evaluate(ob[0:2], ob[2:4])
        #print(en_ex)
        solverTime[i] = time.time() - t0
        #env.render()
        ob, reward, done, info = env.step(action)
        """
        print('state : ', x)
        print('velocity : ', xdot)
        print('forward velocity : ', qdot)
        """
        q[i, 0:3] = ob[0:n]
        #q[i, 3:5] = x_f_t
        #q[i, 5:7] = x_b_t
        #q[i, 7:9] = x_r_t
        #q[i, 9:11] = x_l_t
    qs.append(q)
    solverTimes.append(solverTime)
    ## Plotting the results
    res = {}
    res['qs'] = qs
    res['solverTimes'] = solverTimes
    res['obsts'] = obsts
    res['dt'] = env._dt
    res['x_ee_fun'] = x_ee_fun
    res['goal'] = x_d
    return res

if __name__ == "__main__":
    n_steps = 3000
    res = pointMass(n_steps=n_steps)
    fk_fun = res['x_ee_fun']
    sol_indices = [0]
    robotPlot = RobotPlot([res['qs'][i] for i in sol_indices], fk_fun, 3, types=[4])
    robotPlot.initFig(2, 2)
    robotPlot.addObstacle([0, 1], res['obsts'])
    robotPlot.plot()
    robotPlot.addGoal([0, 1], res['goal'])
    robotPlot.makeAnimation(n_steps)
    solverAxs = robotPlot.getAxs([2, 3])
    solverPlot = SolverPlot([res['solverTimes'][i] for i in sol_indices], 1, 1, axs=solverAxs)
    solverPlot.plot()
    robotPlot.show()
