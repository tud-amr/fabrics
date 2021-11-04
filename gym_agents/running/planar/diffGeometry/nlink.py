import gym
import nLinkReacher
import time
import casadi as ca
import numpy as np

from optFabrics.planner.fabricPlanner import DefaultFabricPlanner
from optFabrics.planner.default_geometries import CollisionGeometry, LimitGeometry, GoalGeometry
from optFabrics.planner.default_energies import CollisionLagrangian, ExecutionLagrangian
from optFabrics.planner.default_maps import CollisionMap, UpperLimitMap, LowerLimitMap
from optFabrics.planner.default_leaves import defaultAttractor

from obstacle import Obstacle
from robotPlot import RobotPlot
from solverPlot import SolverPlot

from casadiFk import casadiFk
from numpyFk import numpyFk


def nlink(n=3, n_steps=5000):
    # setting up the problem
    obsts = [
        Obstacle(np.array([2.0, 4.0]), 0.5),
        Obstacle(np.array([0.0, 3.0]), 0.5),
    ]
    planner = DefaultFabricPlanner(n)
    q, qdot = planner.var()
    fks = []
    for i in range(1, n + 1):
        fks.append(ca.SX(casadiFk(q, i)[0:2]))
    # collision avoidance
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    lag_col = CollisionLagrangian(x, xdot)
    geo_col = CollisionGeometry(x, xdot, exp=3, lam=2)
    for fk in fks:
        for obst in obsts:
            dm_col = CollisionMap(q, qdot, fk, obst.x(), obst.r())
            planner.addGeometry(dm_col, lag_col, geo_col)
    # joint limit avoidance
    lag_lim = CollisionLagrangian(x, xdot)
    geo_lim = LimitGeometry(x, xdot, lam=4.00, exp=2)
    for i in range(n):
        dm_lim_upper = UpperLimitMap(q, qdot, 1.0 * np.pi, i)
        planner.addGeometry(dm_lim_upper, lag_lim, geo_lim)
        dm_lim_lower = LowerLimitMap(q, qdot, -1.0 * np.pi, i)
        planner.addGeometry(dm_lim_lower, lag_lim, geo_lim)
    # forcing term
    q_d = np.array([-3.0, -0.0])
    dm_psi, lag_psi, geo_psi, x_psi, xdot_psi = defaultAttractor(q, qdot, q_d, fk)
    geo_psi = GoalGeometry(x_psi, xdot_psi, k_psi=2)
    planner.addForcingGeometry(dm_psi, lag_psi, geo_psi)
    # execution energy
    exLag = ExecutionLagrangian(q, qdot)
    planner.setExecutionEnergy(exLag)
    # Speed control
    ex_factor = 1.0
    planner.setDefaultSpeedControl(x_psi, dm_psi, exLag, ex_factor, r_b=1.0)
    planner.concretize()
    # setup environment
    # running the simulation
    env = gym.make("nLink-reacher-acc-v0", n=n, dt=0.05)
    qs = []
    solverTimes = []
    print("Starting episode")
    q0 = np.zeros(n)
    q0dot = np.array([0.1, 0.4, 0.1, 0.0, 0.0])
    ob = env.reset(q0, q0dot)
    q = np.zeros((n_steps, n))
    solverTime = np.zeros(n_steps)
    t = 0.0
    for i in range(n_steps):
        t += env._dt
        t0 = time.time()
        action = planner.computeAction(ob[0:n], ob[n : 2 * n])
        solverTime[i] = time.time() - t0
        # env.render()
        ob, reward, done, info = env.step(action)
        q[i, :] = ob[0:n]
    qs.append(q)
    solverTimes.append(solverTime)
    res = {}
    res['qs'] = qs
    res['solverTimes'] = solverTimes
    res['dt'] = env._dt
    res['obsts'] = obsts
    return res

if __name__ == "__main__":
    n_steps = 5000
    n = 5
    res = nlink(n=n, n_steps=n_steps)
    # Plotting the results
    fk_fun = lambda q, n: numpyFk(q, n)[0:3]
    robotPlot = RobotPlot(res['qs'], fk_fun, 2, types=[1], dt=res['dt'])
    robotPlot.initFig(2, 2)
    robotPlot.addObstacle([0, 1], res['obsts'])
    robotPlot.plot()
    robotPlot.makeAnimation(n_steps)
    solverAxs = robotPlot.getAxs([2, 3])
    solverPlot = SolverPlot(res['solverTimes'], 1, 1, axs=solverAxs)
    solverPlot.plot()
    robotPlot.show()
