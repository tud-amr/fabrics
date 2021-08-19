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

from obstacle import Obstacle, DynamicObstacle
from robotPlot import RobotPlot
from solverPlot import SolverPlot


def pointMassDynamicGoal(n_steps=5000):
    env = gym.make("point-robot-acc-v0", dt=0.01)
    t = ca.SX.sym("t", 1)
    w = 1.0
    x_obst = ca.vertcat(0.5, -3.0 * ca.sin(w * t))
    v_obst = ca.jacobian(x_obst, t)
    a_obst = ca.jacobian(v_obst, t)
    x_obst_fun = ca.Function("x_obst_fun", [t], [x_obst])
    v_obst_fun = ca.Function("v_obst_fun", [t], [v_obst])
    a_obst_fun = ca.Function("a_obst_fun", [t], [a_obst])
    r = 1.0
    obsts = [DynamicObstacle(x_obst_fun, r), Obstacle(np.array([-1.0, 0.5]), 0.15)]
    t = ca.SX.sym("t", 1)
    x_d = ca.vertcat(2.0 * ca.cos(w * t), 1.5 * ca.sin(w * t))
    x_goal = ca.Function("x_goal", [t], [x_d])
    v_d = ca.jacobian(x_d, t)
    a_d = ca.jacobian(v_d, t)
    x_d_fun = ca.Function("x_d_fun", [t], [x_d])
    v_d_fun = ca.Function("v_d_fun", [t], [v_d])
    a_d_fun = ca.Function("a_d_fun", [t], [a_d])
    n = 2
    planner = DefaultFabricPlanner(n, m_base=1.0)
    q, qdot = planner.var()
    # collision avoidance
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    fks = [q]
    lag_col = CollisionLagrangian(x, xdot)
    geo_col = CollisionGeometry(x, xdot, exp=3)
    for obst in obsts:
        q_p = ca.SX.sym('q_p', 2)
        qdot_p = ca.SX.sym('qdot_p', 2)
        qddot_p = ca.SX.sym('qddot_p', 2)
        q_rel = ca.SX.sym('q_rel', 2)
        qdot_rel = ca.SX.sym('qdot_rel', 2)
        for fk in fks:
            if isinstance(obst, DynamicObstacle):
                phi_n = ca.norm_2(q_rel) / obst.r()  - 1
                dm_n = DifferentialMap(phi_n, q=q_rel, qdot=qdot_rel)
                dm_rel = RelativeDifferentialMap(q=q, qdot=qdot, q_p=q_p, qdot_p=qdot_p, qddot_p=qddot_p)
                planner.addGeometry(dm_rel, lag_col.pull(dm_n), geo_col.pull(dm_n))
            elif isinstance(obst, Obstacle):
                dm_col = CollisionMap(q, qdot, fk, obst.x(), obst.r())
                planner.addGeometry(dm_col, lag_col, geo_col)
    # forcing term
    dm_psi, lag_psi, geo_psi, x_psi, xdot_psi, xdot_g = defaultDynamicAttractor(
        q, qdot, q
    )
    planner.addForcingGeometry(dm_psi, lag_psi, geo_psi, goalVelocity=xdot_g)
    # execution energy
    exLag = ExecutionLagrangian(q, qdot)
    planner.setExecutionEnergy(exLag)
    # Speed control
    ex_factor = 1.0
    planner.setDefaultSpeedControl(x_psi, dm_psi, exLag, ex_factor, b=[0.4, 10.0])
    # planner.setConstantSpeedControl(beta=5.0)
    planner.concretize()
    # setup environment
    qs = []
    solverTimes = []
    x0s = [np.array([2.3, -1.0 + i * 0.2]) for i in range(2)]
    xdot0s = [np.array([-1.0, -0.0])]
    x0s = [np.array([2.0, 2.0])]
    xdot0s = [np.array([-1.0, -1.0])]
    # running the simulation
    for e in range(2):
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
                    q_p_t = np.array(x_obst_fun(t))[:, 0]
                    qdot_p_t = np.array(v_obst_fun(t))[:, 0]
                    qddot_p_t = np.array(a_obst_fun(t))[:, 0]
                    q_g_t = np.array(x_d_fun(t))[:, 0]
                    qdot_g_t = np.array(v_d_fun(t))[:, 0]
                    qddot_g_t = np.array(a_d_fun(t))[:, 0]
                    if e == 0:
                        qdot_g_t = np.zeros(2)
                        qddot_g_t = np.zeros(2)
                    t0 = time.time()
                    action = planner.computeAction(
                        ob[0:2],
                        ob[2:4],
                        q_p_t,
                        qdot_p_t,
                        qddot_p_t,
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
    res["obsts"] = obsts
    res["dt"] = env._dt
    res["x_d"] = x_goal
    return res


if __name__ == "__main__":
    n_steps = 10000
    res = pointMassDynamicGoal(n_steps=n_steps)
    fk_fun = lambda q: q
    sol_indices = [0, 1]
    robotPlot = RobotPlot(
        [res["qs"][i] for i in sol_indices], fk_fun, 2, types=[0, 0], dt=res["dt"]
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
