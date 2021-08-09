import gym
import nLinkReacher
import time
import casadi as ca
import numpy as np

# fabrics
from optFabrics.planner.fabricPlanner import DefaultFabricPlanner
from optFabrics.planner.default_geometries import CollisionGeometry
from optFabrics.planner.default_energies import CollisionLagrangian, ExecutionLagrangian
from optFabrics.planner.default_maps import CollisionMap, VariableCollisionMap
from optFabrics.planner.default_leaves import defaultAttractor

# robotUtils
from obstacle import DynamicObstacle, Obstacle
from robotPlot import RobotPlot
from solverPlot import SolverPlot
from casadiFk import casadiFk
from numpyFk import numpyFk


def nlinkDynamic(n=3, n_steps=5000):
    # Define the robot
    env = gym.make('nLink-reacher-acc-v0', n=n, dt=0.01)
    # Define the problem
    x_d = np.array([-2.0, -1.0])
    t = ca.SX.sym('t', 1)
    x_obst = ca.vertcat(-1.2 + 0.1 * t**2, 1.3)
    v_obst = ca.jacobian(x_obst, t)
    a_obst = ca.jacobian(v_obst, t)
    x_obst_fun = ca.Function("x_obst_fun", [t], [x_obst])
    v_obst_fun = ca.Function("v_obst_fun", [t], [v_obst])
    a_obst_fun = ca.Function("a_obst_fun", [t], [a_obst])
    x2_obst = ca.vertcat(-1.2, 2.3 - 0.5 * t)
    v2_obst = ca.jacobian(x2_obst, t)
    a2_obst = ca.jacobian(v2_obst, t)
    x2_obst_fun = ca.Function("x2_obst_fun", [t], [x2_obst])
    v2_obst_fun = ca.Function("v2_obst_fun", [t], [v2_obst])
    a2_obst_fun = ca.Function("a2_obst_fun", [t], [a2_obst])
    obsts = [
                DynamicObstacle(x2_obst_fun, 1.0),
                DynamicObstacle(x_obst_fun, 1.0),
                Obstacle(np.array([-0.5, -1.5]), 0.5)
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
    geo_col = CollisionGeometry(x, xdot)
    for obst in obsts:
        q_p = ca.SX.sym('q_p', 2)
        qdot_p = ca.SX.sym('qdot_p', 2)
        qddot_p = ca.SX.sym('qddot_p', 2)
        for fk in fks:
            if isinstance(obst, DynamicObstacle):
                dm_col = VariableCollisionMap(q, qdot, fk, obst.r(), q_p, qdot_p, qddot_p)
            elif isinstance(obst, Obstacle):
                dm_col = CollisionMap(q, qdot, fk, obst.x(), obst.r())
            planner.addGeometry(dm_col, lag_col, geo_col)
    # forcing term
    dm_psi, lag_psi, geo_psi, x_psi, _ = defaultAttractor(q, qdot, x_d, fk)
    planner.addForcingGeometry(dm_psi, lag_psi, geo_psi)
    # execution energy
    exLag = ExecutionLagrangian(q, qdot)
    planner.setExecutionEnergy(exLag)
    # Speed control
    ex_factor = 1.0
    planner.setDefaultSpeedControl(x_psi, dm_psi, exLag, ex_factor, r_b=2)
    planner.concretize()
    # setup environment
    qs = []
    solverTimes = []
    # running the simulation
    #ob = env.reset(np.array([2.0, 1.0]), np.array([0.0, 0.0]))
    ob = env.reset()
    print("Starting episode")
    q = np.zeros((n_steps, n))
    t = 0.0
    solverTime = np.zeros(n_steps)
    for i in range(n_steps):
        t += env._dt
        t0 = time.time()
        q_p_t = np.array(x_obst_fun(t))[:, 0]
        qdot_p_t = np.array(v_obst_fun(t))[:, 0]
        qddot_p_t = np.array(a_obst_fun(t))[:, 0]
        q2_p_t = np.array(x2_obst_fun(t))[:, 0]
        q2dot_p_t = np.array(v2_obst_fun(t))[:, 0]
        q2ddot_p_t = np.array(a2_obst_fun(t))[:, 0]
        action = planner.computeAction(ob[0:n], ob[n:2*n], q_p_t, qdot_p_t, qddot_p_t, q2_p_t, q2dot_p_t, q2ddot_p_t)
        #action = planner.computeAction(ob[0:n], ob[n:2*n], q2_p_t, q2dot_p_t)
        #action = planner.computeAction(ob[0:n], ob[n:2*n])
        solverTime[i] = time.time() - t0
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
    n = 3
    n_steps = 10000
    res = nlinkDynamic(n=n, n_steps=n_steps)
    # Plotting the results
    fk_fun = lambda q, n: numpyFk(q, n)[0:3]
    robotPlot = RobotPlot(res['qs'], fk_fun, 2, types=[1])
    robotPlot.initFig(2, 2)
    robotPlot.addObstacle([0, 1], res['obsts'])
    robotPlot.plot()
    robotPlot.makeAnimation(n_steps)
    solverAxs = robotPlot.getAxs([2, 3])
    solverPlot = SolverPlot(res['solverTimes'], 1, 1, axs=solverAxs)
    solverPlot.plot()
    robotPlot.show()
