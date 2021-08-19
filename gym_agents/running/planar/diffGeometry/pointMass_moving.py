import gym
import pointRobot
import time
import casadi as ca
import numpy as np

from optFabrics.planner.fabricPlanner import DefaultFabricPlanner
from optFabrics.planner.default_geometries import CollisionGeometry, GoalGeometry
from optFabrics.planner.default_energies import CollisionLagrangian, ExecutionLagrangian
from optFabrics.planner.default_maps import CollisionMap
from optFabrics.planner.default_leaves import defaultAttractor

from optFabrics.diffGeometry.diffMap import DifferentialMap, RelativeDifferentialMap

from obstacle import DynamicObstacle, Obstacle
from robotPlot import RobotPlot
from solverPlot import SolverPlot


def pointMassDynamic(n_steps=5000):
    ## setting up the problem
    n = 2
    env = gym.make('point-robot-acc-v0', dt=0.005)
    t = ca.SX.sym('t', 1)
    x_obst = ca.vertcat(0.5 - 0.5 * t, -3.0 + t)
    v_obst = ca.jacobian(x_obst, t)
    a_obst = ca.jacobian(v_obst, t)
    x_obst_fun = ca.Function("x_obst_fun", [t], [x_obst])
    v_obst_fun = ca.Function("v_obst_fun", [t], [v_obst])
    a_obst_fun = ca.Function("a_obst_fun", [t], [a_obst])
    x2_obst = ca.vertcat(-0.5, 2.0 - 0.5 * t)
    v2_obst = ca.jacobian(x2_obst, t)
    a2_obst = ca.jacobian(v2_obst, t)
    x2_obst_fun = ca.Function("x2_obst_fun", [t], [x2_obst])
    v2_obst_fun = ca.Function("v2_obst_fun", [t], [v2_obst])
    a2_obst_fun = ca.Function("a2_obst_fun", [t], [a2_obst])
    r = 1.0
    r2 = 0.5
    obsts = [
                DynamicObstacle(x_obst_fun, r),
                DynamicObstacle(x2_obst_fun, r2),
                Obstacle(np.array([-0.8, 1.0]), 0.5)
            ]
    planner = DefaultFabricPlanner(n, m_base=4)
    q, qdot = planner.var()
    fks = [q]
    # collision avoidance
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    lag_col = CollisionLagrangian(x, xdot)
    geo_col = CollisionGeometry(x, xdot, exp=3, lam=10)
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
    q_d = np.array([-2.0, -1.5])
    dm_psi, lag_psi, _, x_psi, xdot_psi  = defaultAttractor(q, qdot, q_d, fk)
    geo_psi = GoalGeometry(x_psi, xdot_psi, k_psi=5)
    planner.addForcingGeometry(dm_psi, lag_psi, geo_psi)
    # execution energy
    exLag = ExecutionLagrangian(q, qdot)
    planner.setExecutionEnergy(exLag)
    # Speed control
    ex_factor = 2.0
    planner.setDefaultSpeedControl(x_psi, dm_psi, exLag, ex_factor)
    planner.concretize()
    # setup environment
    qs = []
    solverTimes = []
    x0s = [np.array([2.3, -1.0 + i * 0.2]) for i in range(1)]
    xdot0s = [np.array([np.cos(i*np.pi/5), np.sin(i*np.pi/5)]) for i in range(10)]
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
                q_p_t = np.array(x_obst_fun(t))[:, 0]
                qdot_p_t = np.array(x_obst_fun(t))[:, 0]
                qddot_p_t = np.array(a_obst_fun(t))[:, 0]
                q2_p_t = np.array(x2_obst_fun(t))[:, 0]
                q2dot_p_t = np.array(v2_obst_fun(t))[:, 0]
                q2ddot_p_t = np.array(a2_obst_fun(t))[:, 0]
                action = planner.computeAction(ob[0:2], ob[2:4], q_p_t, qdot_p_t, qddot_p_t, q2_p_t ,q2dot_p_t, q2ddot_p_t)
                #_, _, en_ex = exLag.evaluate(ob[0:2], ob[2:4])
                #print(en_ex)
                solverTime[i] = time.time() - t0
                # env.render()
                ob, reward, done, info = env.step(action)
                q[i, :] = ob[0:n]
            qs.append(q)
            solverTimes.append(solverTime)
    ## Plotting the results
    res = {}
    res['qs'] = qs
    res['solverTimes'] = solverTimes
    res['obsts'] = obsts
    res['dt'] = env._dt
    return res

if __name__ == "__main__":
    n_steps = 5000
    res = pointMassDynamic(n_steps=n_steps)
    fk_fun = lambda q : q
    sol_indices = [0, 3, 7, 9]
    robotPlot = RobotPlot([res['qs'][i] for i in sol_indices], fk_fun, 2, types=[0, ] * len(sol_indices))
    robotPlot.initFig(2, 2)
    robotPlot.addObstacle([0, 1, 2, 3], res['obsts'])
    robotPlot.plot()
    robotPlot.makeAnimation(n_steps)
    robotPlot.addSolutions(0, res['qs'])
    robotPlot.show()
