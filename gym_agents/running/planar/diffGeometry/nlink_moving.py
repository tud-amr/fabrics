import gym
import nLinkReacher
import time
import casadi as ca
import numpy as np

# fabrics
from optFabrics.planner.fabricPlanner import DefaultFabricPlanner
from optFabrics.planner.default_geometries import CollisionGeometry
from optFabrics.planner.default_energies import CollisionLagrangian, ExecutionLagrangian
from optFabrics.planner.default_maps import CollisionMap
from optFabrics.planner.default_leaves import defaultAttractor

from optFabrics.diffGeometry.diffMap import DifferentialMap, RelativeDifferentialMap
from optFabrics.diffGeometry.referenceTrajectory import ReferenceTrajectory

# robotUtils
from obstacle import DynamicObstacle, Obstacle
from robotPlot import RobotPlot
from solverPlot import SolverPlot
from casadiFk import casadiFk
from numpyFk import numpyFk


def nlinkDynamic(n=3, n_steps=5000):
    # Define the robot
    env = gym.make('nLink-reacher-acc-v0', n=n, dt=0.010)
    # Define the problem
    x_d = np.array([-2.0, 1.0])
    t = ca.SX.sym('t', 1)
    x_obst = ca.vertcat(-1.2 + 0.4 * t**1, 1.3)
    x_obst = ca.vertcat(7.2 - 0.1 * t**2, -2.5 + 0.5 * t)
    x_obst_fun = ca.Function("x_obst_fun", [t], [x_obst])
    refTraj_obst1 = ReferenceTrajectory(2, ca.SX(np.identity(2)), traj=x_obst, t=t, name="obst1")
    refTraj_obst1.concretize()
    x2_obst = ca.vertcat(-1.2, 2.3 - 0.7 * t)
    x2_obst_fun = ca.Function("x2_obst_fun", [t], [x2_obst])
    refTraj_obst2 = ReferenceTrajectory(2, ca.SX(np.identity(2)), traj=x2_obst, t=t, name="obst2")
    refTraj_obst2.concretize()
    obsts = [
                DynamicObstacle(x2_obst_fun, 1.0),
                DynamicObstacle(x_obst_fun, 1.0),
                #Obstacle(np.array([-0.5, -1.5]), 0.5)
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
    geo_col = CollisionGeometry(x, xdot, exp=3, lam=20)
    for i, obst in enumerate(obsts):
        x_col = ca.SX.sym("x_col", 2)
        xdot_col = ca.SX.sym("xdot_col", 2)
        x_rel = ca.SX.sym("x_rel", 2)
        xdot_rel = ca.SX.sym("xdot_rel", 2)
        for fk in fks:
            if isinstance(obst, DynamicObstacle):
                if i == 1:
                    refTraj = refTraj_obst1
                elif i == 0:
                    refTraj = refTraj_obst2
                phi_n = ca.norm_2(x_rel)/ obst.r() - 1
                dm_n = DifferentialMap(phi_n, q=x_rel, qdot=xdot_rel)
                dm_rel = RelativeDifferentialMap(q=x_col, qdot=xdot_col, refTraj=refTraj)
                dm_col = DifferentialMap(fk, q=q, qdot=qdot)
                planner.addGeometry(dm_col, lag_col.pull(dm_n).pull(dm_rel), geo_col.pull(dm_n).pull(dm_rel))
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
    planner.setDefaultSpeedControl(x_psi, dm_psi, exLag, ex_factor, r_b=1.0)
    planner.concretize()
    # setup environment
    qs = []
    solverTimes = []
    # running the simulation
    ob = env.reset()
    print("Starting episode")
    q = np.zeros((n_steps, n))
    t = 0.0
    solverTime = np.zeros(n_steps)
    for i in range(n_steps):
        if i % 1000 == 0:
            print("time step : %d" % i)
        t += env._dt
        t0 = time.time()
        q_p_t, qdot_p_t, qddot_p_t = refTraj_obst1.evaluate(t)
        q2_p_t, q2dot_p_t, q2ddot_p_t = refTraj_obst2.evaluate(t)
        action = planner.computeAction(ob[0:n], ob[n:2*n], q_p_t, qdot_p_t, qddot_p_t, q2_p_t, q2dot_p_t, q2ddot_p_t)
        solverTime[i] = time.time() - t0
        ob, reward, done, info = env.step(action)
        q[i, :] = ob[0:n]
    qs.append(q)
    solverTimes.append(solverTime)
    res = {}
    res['qs'] = qs
    res['solverTimes'] = solverTimes
    res['goal'] = x_d
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
    robotPlot.addGoal([0, 1], res['goal'])
    robotPlot.plot()
    robotPlot.makeAnimation(n_steps)
    solverAxs = robotPlot.getAxs([2, 3])
    solverPlot = SolverPlot(res['solverTimes'], 1, 1, axs=solverAxs)
    solverPlot.plot()
    robotPlot.show()
