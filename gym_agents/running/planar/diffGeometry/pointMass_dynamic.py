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
from optFabrics.diffGeometry.referenceTrajectory import ReferenceTrajectory

from obstacle import Obstacle, DynamicObstacle
from robotPlot import RobotPlot
from solverPlot import SolverPlot

# requires Jdot_sign to be set to -1!?


def pointMassDynamicGoal(n_steps=5000):
    env = gym.make("point-robot-acc-v0", dt=0.01)
    t = ca.SX.sym("t", 1)
    w = 1.0
    x_obst = ca.vertcat(0.5, -3.0 * ca.sin(w * t))
    x_obst_fun = ca.Function("x_obst_fun", [t], [x_obst])
    refTraj_obst = ReferenceTrajectory(2, ca.SX(np.identity(2)), traj=x_obst, t=t, name="obst")
    refTraj_obst.concretize()
    r = 1.0
    obsts = [
        DynamicObstacle(x_obst_fun, r),
        Obstacle(np.array([-1.0, 0.5]), 0.15)
    ]
    x_d = ca.vertcat(2.0 * ca.cos(w * t), 1.5 * ca.sin(w * t))
    x_goal = ca.Function("x_goal", [t], [x_d])
    refTraj_goal = ReferenceTrajectory(2, ca.SX(np.identity(2)), traj=x_d, t=t, name="goal")
    refTraj_goal.concretize()
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
        q_rel = ca.SX.sym('q_rel', 2)
        qdot_rel = ca.SX.sym('qdot_rel', 2)
        for fk in fks:
            if isinstance(obst, DynamicObstacle):
                phi_n = ca.norm_2(q_rel) / obst.r()  - 1
                dm_n = DifferentialMap(phi_n, q=q_rel, qdot=qdot_rel)
                dm_rel = RelativeDifferentialMap(q=q, qdot=qdot, refTraj=refTraj_obst)
                planner.addGeometry(dm_rel, lag_col.pull(dm_n), geo_col.pull(dm_n))
            elif isinstance(obst, Obstacle):
                dm_col = CollisionMap(q, qdot, fk, obst.x(), obst.r())
                planner.addGeometry(dm_col, lag_col, geo_col)
    # forcing term
    dm_psi, lag_psi, geo_psi, x_psi, xdot_psi = defaultDynamicAttractor(
        q, qdot, q, refTraj_goal, k_psi=15.0
    )
    planner.addForcingGeometry(dm_psi, lag_psi, geo_psi, goalVelocity=refTraj_goal.xdot())
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
                    q_p_t, qdot_p_t, qddot_p_t = refTraj_obst.evaluate(t)
                    q_g_t, qdot_g_t, qddot_g_t = refTraj_goal.evaluate(t)
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
