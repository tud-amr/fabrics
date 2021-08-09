import gym
import nLinkReacher
import time
import casadi as ca
import numpy as np

from optFabrics.planner.fabricPlanner import DefaultFabricPlanner
from optFabrics.planner.default_geometries import CollisionGeometry
from optFabrics.planner.default_energies import (
    CollisionLagrangian,
    ExecutionLagrangian,
)
from optFabrics.planner.default_maps import (
    CollisionMap,
    VariableCollisionMap,
)
from optFabrics.planner.default_leaves import defaultDynamicAttractor

from obstacle import Obstacle, DynamicObstacle
from robotPlot import RobotPlot
from solverPlot import SolverPlot
from casadiFk import casadiFk
from numpyFk import numpyFk


def nlinkDynamicGoal(n=3, n_steps=5000):
    env = gym.make("nLink-reacher-acc-v0", n=n, dt=0.01)
    t = ca.SX.sym("t", 1)
    w_obst = 0.3
    x_obst = ca.vertcat(5 * ca.sin(w_obst * t), -3.0)
    v_obst = ca.jacobian(x_obst, t)
    x_obst_fun = ca.Function("x_obst_fun", [t], [x_obst])
    v_obst_fun = ca.Function("v_obst_fun", [t], [v_obst])
    r = 1.0
    obsts = [DynamicObstacle(x_obst_fun, r), Obstacle(np.array([-1.0, 2.5]), 0.5)]
    # goal
    t = ca.SX.sym("t", 1)
    w = 1.0
    x_d = ca.vertcat(1.5 + 0.7 * ca.sin(w * t), -1 + 1 * ca.cos(w * t))
    x_goal = ca.Function("x_goal", [t], [x_d])
    v_d = ca.jacobian(x_d, t)
    a_d = ca.jacobian(v_d, t)
    x_d_fun = ca.Function("x_d_fun", [t], [x_d])
    v_d_fun = ca.Function("v_d_fun", [t], [v_d])
    a_d_fun = ca.Function("a_d_fun", [t], [a_d])
    planner = DefaultFabricPlanner(n, m_base=0.1)
    q, qdot = planner.var()
    # collision avoidance
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    lag_col = CollisionLagrangian(x, xdot)
    geo_col = CollisionGeometry(x, xdot)
    fks = []
    for i in range(1, n + 1):
        fks.append(ca.SX(casadiFk(q, i)[0:2]))
    for obst in obsts:
        q_p = ca.SX.sym("q_p", 2)
        qdot_p = ca.SX.sym("qdot_p", 2)
        qddot_p = ca.SX.sym("qddot_p", 2)
        for fk in fks:
            if isinstance(obst, DynamicObstacle):
                dm_col = VariableCollisionMap(
                    q, qdot, fk, obst.r(), q_p, qdot_p, qddot_p
                )
            elif isinstance(obst, Obstacle):
                dm_col = CollisionMap(q, qdot, fk, obst.x(), obst.r())
            planner.addGeometry(dm_col, lag_col, geo_col)
    # forcing term
    dm_psi, lag_psi, geo_psi, x_psi, xdot_psi, xdot_g = defaultDynamicAttractor(
        q, qdot, fks[-1]
    )
    planner.addForcingGeometry(dm_psi, lag_psi, geo_psi, goalVelocity=xdot_g)
    # execution energy
    exLag = ExecutionLagrangian(q, qdot)
    planner.setExecutionEnergy(exLag)
    # Speed control
    ex_factor = 1.0
    planner.setDefaultSpeedControl(x_psi, dm_psi, exLag, ex_factor, b=[2.0, 5.0])
    # planner.setConstantSpeedControl(beta=2.5)
    planner.concretize()
    # setup environment
    qs = []
    solverTimes = []
    # running the simulation
    for e in range(2):
        ob = env.reset()
        print("Starting episode")
        q = np.zeros((n_steps, n))
        t = 0.0
        solverTime = np.zeros(n_steps)
        for i in range(n_steps):
            if i % 1000 == 0:
                print("time step : ", i)
            t += env._dt
            q_p_t = np.array(x_obst_fun(t))[:, 0]
            qdot_p_t = np.array(v_obst_fun(t))[:, 0]
            qddot_p_t = np.array(v_obst_fun(t))[:, 0]
            q_g_t = np.array(x_d_fun(t))[:, 0]
            qdot_g_t = np.array(v_d_fun(t))[:, 0]
            qddot_g_t = np.array(a_d_fun(t))[:, 0]
            if e == 1:
                if i == 0:
                    print("run %i with no relative part" % e)
                qdot_g_t = np.zeros(2)
                qddot_g_t = np.zeros(2)
            t0 = time.time()
            # action = planner.computeAction(ob[0:n], ob[n:2*n], q_p_t, qdot_p_t, q_g_t, qdot_g_t)
            q_t = ob[0:n]
            qdot_t = ob[n : 2 * n]
            action = planner.computeAction(
                q_t, qdot_t, q_p_t, qdot_p_t, qddot_p_t, q_g_t, qdot_g_t, qddot_g_t
            )
            # action = planner.computeAction(ob[0:n], ob[n:2*n])
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
    res["obsts"] = obsts
    res["dt"] = env._dt
    res["x_d"] = x_goal
    return res


if __name__ == "__main__":
    n_steps = 2000
    n = 3
    res = nlinkDynamicGoal(n=n, n_steps=n_steps)
    fk_fun = lambda q, n: numpyFk(q, n)[0:3]
    sol_indices = [0, 1]
    robotPlot = RobotPlot(
        [res["qs"][i] for i in sol_indices], fk_fun, 2, types=[1, 1], dt=res["dt"]
    )
    robotPlot.initFig(2, 2)
    robotPlot.addObstacle([0, 1], res["obsts"])
    robotPlot.plot()
    robotPlot.makeAnimation(n_steps)
    robotPlot.addGoal([0, 1], res["x_d"])
    solverAxs = robotPlot.getAxs([2, 3])
    solverPlot = SolverPlot(
        [res["solverTimes"][i] for i in sol_indices], 1, 1, axs=solverAxs
    )
    solverPlot.plot()
    robotPlot.show()
