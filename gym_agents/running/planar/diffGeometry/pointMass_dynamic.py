import gym
import pointRobot
import time
import casadi as ca
import numpy as np

from optFabrics.planner.fabricPlanner import DefaultFabricPlanner
from optFabrics.planner.default_geometries import CollisionGeometry, GoalGeometry
from optFabrics.planner.default_energies import CollisionLagrangian, ExecutionLagrangian, GoalLagrangian
from optFabrics.planner.default_maps import CollisionMap, VariableGoalMap, VariableCollisionMap
from optFabrics.planner.default_leaves import defaultAttractor


from obstacle import Obstacle, DynamicObstacle
from robotPlot import RobotPlot
from solverPlot import SolverPlot


def pointMassDynamicGoal(n_steps=5000):
    env = gym.make('point-robot-acc-v0', dt=0.01)
    t = ca.SX.sym('t', 1)
    w_obst = 0.2
    x_obst = ca.vertcat(0.5, -3.0 * ca.sin(w_obst * t))
    v_obst = ca.jacobian(x_obst, t)
    x_obst_fun = ca.Function("x_obst_fun", [t], [x_obst])
    v_obst_fun = ca.Function("v_obst_fun", [t], [x_obst])
    v2_obst = np.array([-0.0, -0.5])
    x2_obst = ca.vertcat(-0.5, 2.0) + t * v2_obst
    x2_obst_fun = ca.Function("x2_obst_fun", [t], [x2_obst])
    r = 1.0
    r2 = 0.5
    obsts = [
                DynamicObstacle(x_obst_fun, r),
                Obstacle(np.array([-1.0, 0.5]), 0.5)
            ]
    n = 2
    planner = DefaultFabricPlanner(n, m_base=1.0)
    q, qdot = planner.var()
    # collision avoidance
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    lag_col = CollisionLagrangian(x, xdot)
    geo_col = CollisionGeometry(x, xdot)
    fks = [q]
    for obst in obsts:
        q_p = ca.SX.sym('q_p', 2)
        qdot_p = ca.SX.sym('qdot_p', 2)
        for fk in fks:
            if isinstance(obst, DynamicObstacle):
                dm_col = VariableCollisionMap(q, qdot, fk, obst.r(), q_p, qdot_p)
            elif isinstance(obst, Obstacle):
                dm_col = CollisionMap(q, qdot, fk, obst.x(), obst.r())
            planner.addGeometry(dm_col, lag_col, geo_col)
    # forcing term
    t = ca.SX.sym("t", 1)
    w = 1.0
    x_d = ca.vertcat(2.0 * ca.cos(w * t), 1.5 * ca.sin(w * t))
    x_goal = ca.Function('x_goal', [t], [x_d])
    v_d = ca.jacobian(x_d, t)
    x_d_fun = ca.Function("x_d_fun", [t], [x_d])
    v_d_fun = ca.Function("v_d_fun", [t], [v_d])
    q_d = np.array([-2.0, -1.0])
    dm_psi, lag_psi, _, x_psi, xdot_psi  = defaultAttractor(q, qdot, q_d, fks[-1])
    q_g = ca.SX.sym("q_g", 2)
    qdot_g = ca.SX.sym("qdot_g", 2)
    dm_psi = VariableGoalMap(q, qdot, fks[-1], q_g, qdot_g)
    geo_psi = GoalGeometry(x_psi, xdot_psi, k_psi=5)
    planner.addForcingGeometry(dm_psi, lag_psi, geo_psi)
    # execution energy
    exLag = ExecutionLagrangian(q, qdot)
    planner.setExecutionEnergy(exLag)
    # Speed control
    ex_factor = 5.0
    planner.setDefaultSpeedControl(x_psi, dm_psi, exLag, ex_factor)
    #planner.setConstantSpeedControl(beta=5.0)
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
                        print('time step : ', i)
                    t += env._dt
                    q_p_t = np.array(x_obst_fun(t))[:, 0]
                    qdot_p_t = np.array(v_obst_fun(t))[:, 0]
                    q_g_t = np.array(x_d_fun(t))[:, 0]
                    qdot_g_t = np.array(v_d_fun(t))[:, 0]
                    if e == 0:
                        qdot_g_t = np.zeros(2)
                    t0 = time.time()
                    action = planner.computeAction(ob[0:2], ob[2:4], q_p_t, qdot_p_t, q_g_t, qdot_g_t)
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
    res['x_d'] = x_goal
    return res


if __name__ == "__main__":
    n_steps = 10000
    res = pointMassDynamicGoal(n_steps=n_steps)
    fk_fun = lambda q : q
    sol_indices = [0, 1]
    robotPlot = RobotPlot([res['qs'][i] for i in sol_indices], fk_fun, 2, types=[0, 0], dt=res['dt'])
    robotPlot.initFig(2, 2)
    robotPlot.addObstacle([0, 1], res['obsts'])
    robotPlot.plot()
    robotPlot.makeAnimation(n_steps)
    #robotPlot.addSolutions(0, res['qs'])
    robotPlot.addGoal([0, 1], res['x_d'])
    solverAxs = robotPlot.getAxs([2, 3])
    solverPlot = SolverPlot([res['solverTimes'][i] for i in sol_indices], 1, 1, axs=solverAxs)
    solverPlot.plot()
    robotPlot.show()
