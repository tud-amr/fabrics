import gym
import groundRobots
import time
import casadi as ca
import numpy as np

from optFabrics.planner.nonHolonomicPlanner import (
    DefaultNonHolonomicPlanner,
)
from optFabrics.diffGeometry.diffMap import DifferentialMap, RelativeDifferentialMap
from optFabrics.diffGeometry.referenceTrajectory import AnalyticTrajectory
from optFabrics.diffGeometry.energy import Lagrangian
from optFabrics.planner.default_leaves import defaultAttractor
from optFabrics.planner.default_geometries import CollisionGeometry, GoalGeometry
from optFabrics.planner.default_maps import CollisionMap
from optFabrics.planner.default_energies import (
    CollisionLagrangian,
    ExecutionLagrangian,
)

from MotionPlanningEnv.sphereObstacle import SphereObstacle
from obstacle import Obstacle, DynamicObstacle
from robotPlot import RobotPlot
from solverPlot import SolverPlot


def pointMass(n_steps=5000):
    # setting up the problem
    nx = 3
    nu = 2
    t = ca.SX.sym("t", 1)
    x_obst = ca.vertcat(300.0, 6.0 * ca.sin(0.1 * t))
    x_obst_fun = ca.Function("x_obst_fun", [t], [x_obst])
    refTraj_obst = AnalyticTrajectory(
        2, ca.SX(np.identity(2)), traj=x_obst, t=t, name="obst"
    )
    refTraj_obst.concretize()
    r = 1.0
    obsts = [
        Obstacle(np.array([0.0, -1.0]), 1.0),
        #DynamicObstacle(x_obst_fun, r),
    ]
    planner = DefaultNonHolonomicPlanner(nx, m_base=1.0)
    x, xdot, qdot = planner.vars()
    """
    x_ee = (
        x[0:2]
        + 0.2 * ca.vertcat(ca.cos(x[2]), ca.sin(x[2]))
        + 1.0 * ca.vertcat(ca.cos(x[2] + x[3]), ca.sin(x[2] + x[3]))
    )
    """
    x_ee = x[0:2] + 0.8 *  ca.vertcat(ca.cos(x[2]), ca.sin(x[2]))
    x_ee_fun = ca.Function("x_ee", [x], [x_ee])
    # collision avoidance
    l_front = 0.800
    l_back = -0.800
    l_side = 0.35
    l_i = 0.3
    x_i = x[0:2] + ca.vertcat(l_i * ca.cos(x[2]), l_i * ca.sin(x[2]))
    x_f = x[0:2] + ca.vertcat(l_front * ca.cos(x[2]), l_front * ca.sin(x[2]))
    x_b = x[0:2] + ca.vertcat(l_back * ca.cos(x[2]), l_back * ca.sin(x[2]))
    x_r = x_i + ca.vertcat(l_side * ca.sin(x[2]), -l_side * ca.cos(x[2]))
    x_l = x_i + ca.vertcat(l_side * -ca.sin(x[2]), l_side * ca.cos(x[2]))
    x_col = ca.SX.sym("x_col", 1)
    xdot_col = ca.SX.sym("xdot_col", 1)
    lag_col = CollisionLagrangian(x_col, xdot_col)
    geo_col = CollisionGeometry(x_col, xdot_col)
    fks = [x_f, x_b, x_r, x_l, x_ee]
    fks = [x_ee]
    for obst in obsts:
        x_rel = ca.SX.sym("x_rel", 2)
        xdot_rel = ca.SX.sym("xdot_rel", 2)
        x_col2 = ca.SX.sym("x_col2", 2)
        xdot_col2 = ca.SX.sym("xdot_col2", 2)
        for fk in fks:
            if isinstance(obst, DynamicObstacle):
                phi_n = ca.norm_2(x_rel) / obst.r() - 1
                dm_n = DifferentialMap(phi_n, q=x_rel, qdot=xdot_rel)
                dm_rel = RelativeDifferentialMap(
                    q=x_col2, qdot=xdot_col2, refTraj=refTraj_obst
                )
                dm_col = DifferentialMap(fk, q=x, qdot=xdot)
                planner.addGeometry(
                    dm_col,
                    lag_col.pull(dm_n).pull(dm_rel),
                    geo_col.pull(dm_n).pull(dm_rel),
                )
            elif isinstance(obst, Obstacle):
                dm_col = CollisionMap(x, xdot, fk, obst.x(), obst.r())
                planner.addGeometry(dm_col, lag_col, geo_col)
    # forcing
    fk_ee = x_ee[0:2]
    x_d = np.array([2.0, -2.0])
    x_psi = ca.SX.sym("x_psi", 2)
    dm_psi, lag_psi, _, x_psi, xdot_psi = defaultAttractor(x, xdot, x_d, fk_ee)
    geo_psi = GoalGeometry(x_psi, xdot_psi, k_psi=5)
    planner.addForcingGeometry(dm_psi, lag_psi, geo_psi)
    # finalize
    exLag = ExecutionLagrangian(x, xdot)
    x_ex = x
    xdot_ex = xdot
    exLag = ExecutionLagrangian(x_ex, xdot_ex)
    planner.setExecutionEnergy(exLag)
    exLag.concretize()
    # Speed control
    ex_factor = 1.0
    planner.setDefaultSpeedControl(
        x_psi, dm_psi, exLag, ex_factor,
        #r_b=0.5, b=[0.2, 15.0]
    )
    planner.concretize()
    # setup environment
    qs = []
    solverTimes = []
    # running the simulation
    env = gym.make("ground-robot-acc-v0", dt=0.010, render=True)
    ob = env.reset(pos=np.array([-5.0, 0.0, 0.0]), vel=np.array([1.0, 0.0]))
    contentDict = {'dim': 2, 'type': 'sphere', 'geometry': {'position': obst.x(), 'radius': obst.r()}}
    obstScene = SphereObstacle(name="obst1", contentDict=contentDict)
    env.addObstacle(obstScene)
    print("Starting episode")
    q = np.zeros((n_steps, nx))
    t = 0.0
    solverTime = np.zeros(n_steps)
    for i in range(n_steps):
        if i % 1000 == 0:
            print("time step : ", i)
        t += env._dt
        t0 = time.time()
        x = ob[0:nx]
        xdot = np.concatenate((ob[-nu:], ob[-nx:-nu]))
        qdot = ob[nx : nx + nu]
        q_p_t, qdot_p_t, qddot_p_t = refTraj_obst.evaluate(t)
        action = planner.computeAction(x, xdot,
            #q_p_t, qdot_p_t, qddot_p_t,
            qdot)
        #_, _, en_ex = exLag.evaluate(ob[0:nx], ob[3:2*nx])
        #print(en_ex)
        solverTime[i] = time.time() - t0
        # env.render()
        ob, reward, done, info = env.step(action)
        q[i, 0:nx] = ob[0:nx]
    qs.append(q)
    solverTimes.append(solverTime)
    # Plotting the results
    res = {}
    res["qs"] = qs
    res["solverTimes"] = solverTimes
    res["obsts"] = obsts
    res["dt"] = env._dt
    res["x_ee_fun"] = x_ee_fun
    return res


if __name__ == "__main__":
    n_steps = 70000
    res = pointMass(n_steps=n_steps)
    fk_fun = res["x_ee_fun"]
    sol_indices = [0]
    robotPlot = RobotPlot([res["qs"][i] for i in sol_indices], fk_fun, 2, types=[4])
    lim = 8
    robotPlot.initFig(2, 1, lims=[(-lim, lim), (-lim, lim)])
    robotPlot.addObstacle([0], res["obsts"])
    robotPlot.plot()
    robotPlot.makeAnimation(n_steps)
    solverAxs = robotPlot.getAxs([1])
    solverPlot = SolverPlot(
        [res["solverTimes"][i] for i in sol_indices], 1, 1, axs=solverAxs
    )
    solverPlot.plot()
    robotPlot.show()
