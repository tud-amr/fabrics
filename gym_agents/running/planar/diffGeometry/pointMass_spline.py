import gym
import pointRobot
import time
import casadi as ca
import numpy as np

from optFabrics.planner.fabricPlanner import DefaultFabricPlanner
from optFabrics.planner.default_geometries import CollisionGeometry
from optFabrics.planner.default_energies import CollisionLagrangian, ExecutionLagrangian
from optFabrics.planner.default_maps import CollisionMap
from optFabrics.planner.default_leaves import defaultDynamicAttractor

from optFabrics.diffGeometry.diffMap import DifferentialMap, RelativeDifferentialMap
from optFabrics.diffGeometry.splineTrajectory import SplineTrajectory
from optFabrics.diffGeometry.referenceTrajectory import AnalyticTrajectory

from obstacle import Obstacle, DynamicObstacle
from robotPlot import RobotPlot
from solverPlot import SolverPlot

# requires Jdot_sign to be set to -1!?


def pointMassSplineGoal(n_steps=5000):
    env = gym.make("point-robot-acc-v0", dt=0.01)
    t = ca.SX.sym("t", 1)
    ctrlpts = [[-4.0, -4.0], [-4.0, 4.0], [2.0, 4.0], [3.0, 2.0], [4.0, -4.0]]
    refTraj_goal = SplineTrajectory(2, ca.SX(np.identity(2)), degree=2, ctrlpts=ctrlpts, duration=20)
    n = 2
    planner = DefaultFabricPlanner(n, m_base=1.0)
    q, qdot = planner.var()
    obsts = [
                Obstacle(np.array([0.0, 2.4]), 1.0),
                Obstacle(np.array([-4.0, 1.0]), 0.5),
            ]
    # collision avoidance
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    lag_col = CollisionLagrangian(x, xdot)
    geo_col = CollisionGeometry(x, xdot, exp=3.0, lam=1.0)
    fks = [q]
    for fk in fks:
        for obst in obsts:
            dm_col = CollisionMap(q, qdot, fk, obst.x(), obst.r())
            planner.addGeometry(dm_col, lag_col, geo_col)
    # forcing term
    dm_psi, lag_psi, geo_psi, x_psi, xdot_psi = defaultDynamicAttractor(
        q, qdot, q, refTraj_goal, k_psi=15
    )
    planner.addForcingGeometry(dm_psi, lag_psi, geo_psi, goalVelocity=refTraj_goal.xdot())
    # execution energy
    exLag = ExecutionLagrangian(q, qdot)
    planner.setExecutionEnergy(exLag)
    # Speed control
    ex_factor = 1.0
    planner.setDefaultSpeedControl(x_psi, dm_psi, exLag, ex_factor, b=[2.0, 15.0], r_b=0.01)
    # planner.setConstantSpeedControl(beta=1.0)
    planner.concretize()
    # setup environment
    qs = []
    solverTimes = []
    x0s = [np.array([2.3, -1.0 + i * 0.2]) for i in range(2)]
    xdot0s = [np.array([-1.0, -0.0])]
    x0s = [np.array([-5.0, -5.0])]
    xdot0s = [np.array([-0.0, -0.0])]
    # running the simulation
    for e in range(3):
        for xdot0 in xdot0s:
            for x0 in x0s:
                ob = env.reset(x0, xdot0)
                print("Starting episode")
                q = np.zeros((n_steps, n))
                t = 0.0
                solverTime = np.zeros(n_steps)
                for i in range(n_steps):
                    if i % 1000 == 0:
                        print("time step : ", i)
                    t += env._dt
                    q_g_t, qdot_g_t, qddot_g_t = refTraj_goal.evaluate(t)
                    if e >= 1:
                        qddot_g_t = np.zeros(2)
                    if e == 2:
                        qdot_g_t = np.zeros(2)
                    t0 = time.time()
                    action = planner.computeAction(
                        ob[0:2],
                        ob[2:4],
                        q_g_t,
                        qdot_g_t,
                        qddot_g_t,
                    )
                    # _, _, en_ex = exLag.evaluate(ob[0:2], ob[2:4])
                    # print(en_ex)
                    solverTime[i] = time.time() - t0
                    # env.render()
                    ob, reward, done, info = env.step(action)
                    q[i, :] = ob[0:n]
                qs.append(q)
                solverTimes.append(solverTime)
    ## Plotting the results
    res = {}
    res["qs"] = qs
    res["solverTimes"] = solverTimes
    res["dt"] = env._dt
    res['goal'] = refTraj_goal
    res['obsts'] = obsts
    return res


if __name__ == "__main__":
    n_steps = 10000
    res = pointMassSplineGoal(n_steps=n_steps)
    fk_fun = lambda q: q
    sol_indices = [0, 1, 2]
    robotPlot = RobotPlot(
        [res["qs"][i] for i in sol_indices], fk_fun, 2, types=[0, 0, 0], dt=res["dt"]
    )
    robotPlot.initFig(2, 2)
    robotPlot.plot()
    robotPlot.makeAnimation(n_steps)
    robotPlot.addObstacle([0, 1, 2], res['obsts'])
    robotPlot.addGoal([0, 1, 2], res['goal'])
    robotPlot.addSpline([0, 1, 2], res['goal'].crv())
    robotPlot.addTitles([0, 1, 2], ['dynamic fabric', 'dynamic_fabric without accelerations', 'pseudo_static fabric'])
    """
    solverAxs = robotPlot.getAxs([2, 3])
    solverPlot = SolverPlot(
        [res["solverTimes"][i] for i in sol_indices], 1, 1, axs=solverAxs
    )
    solverPlot.plot()
    """
    robotPlot.show()
