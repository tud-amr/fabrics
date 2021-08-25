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
)
from optFabrics.planner.default_leaves import defaultDynamicAttractor
from optFabrics.diffGeometry.diffMap import DifferentialMap, RelativeDifferentialMap
from optFabrics.diffGeometry.referenceTrajectory import ReferenceTrajectory

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
    x_obst_fun = ca.Function("x_obst_fun", [t], [x_obst])
    refTraj_obst = ReferenceTrajectory(2, ca.SX(np.identity(2)), traj=x_obst, t=t, name="obst")
    refTraj_obst.concretize()
    r = 1.0
    obsts = [
        DynamicObstacle(x_obst_fun, r),
        Obstacle(np.array([-1.0, 2.5]), 0.5), 
    ]
    #obsts = []
    # goal
    w = 1.0
    x_d = ca.vertcat(1.5 + 0.7 * ca.sin(w * t), -1 + 1 * ca.cos(w * t))
    x_goal = ca.Function("x_goal", [t], [x_d])
    refTraj_goal = ReferenceTrajectory(2, ca.SX(np.identity(2)), traj=x_d, t=t, name='goal')
    refTraj_goal.concretize()
    planner = DefaultFabricPlanner(n, m_base=0.1)
    q, qdot = planner.var()
    # collision avoidance
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    print("lag_col_default")
    lag_col = CollisionLagrangian(x, xdot)
    geo_col = CollisionGeometry(x, xdot, exp=1)
    fks = []
    for i in range(2, n+1):
        fks.append(ca.SX(casadiFk(q, i)[0:2]))
    for obst in obsts:
        x_col = ca.SX.sym("x_col", 2)
        xdot_col = ca.SX.sym("xdot_col", 2)
        x_rel = ca.SX.sym("x_rel", 2)
        xdot_rel = ca.SX.sym("xdot_rel", 2)
        for fk in fks:
            if isinstance(obst, DynamicObstacle):
                phi_n = ca.norm_2(x_rel)/ obst.r() - 1
                dm_n = DifferentialMap(phi_n, q=x_rel, qdot=xdot_rel)
                dm_rel = RelativeDifferentialMap(q=x_col, qdot=xdot_col, refTraj=refTraj_obst)
                dm_col = DifferentialMap(fk, q=q, qdot=qdot)
                print('add dynamic Geometry and three pulls')
                planner.addGeometry(dm_col, lag_col.pull(dm_n).pull(dm_rel), geo_col.pull(dm_n).pull(dm_rel))
                print('dynamic geometry done')
            elif isinstance(obst, Obstacle):
                dm_col = CollisionMap(q, qdot, fk, obst.x(), obst.r())
                planner.addGeometry(dm_col, lag_col, geo_col)
    print('col done')
    # forcing term
    dm_psi, lag_psi, geo_psi, x_psi, xdot_psi = defaultDynamicAttractor(
        q, qdot, fks[-1], refTraj_goal, k_psi=20.0
    )
    planner.addForcingGeometry(dm_psi, lag_psi, geo_psi, goalVelocity=refTraj_goal.xdot())
    # execution energy
    exLag = ExecutionLagrangian(q, qdot)
    exLag.concretize()
    planner.setExecutionEnergy(exLag)
    # Speed control
    ex_factor = 1.0
    planner.setDefaultSpeedControl(x_psi, dm_psi, exLag, ex_factor, b=[2.0, 5.0])
    #planner.setConstantSpeedControl(beta=2.5)
    planner.concretize()
    # setup environment
    qs = []
    solverTimes = []
    energies = []
    # running the simulation
    for e in range(2):
        ob = env.reset()
        print("Starting episode")
        q = np.zeros((n_steps, n))
        t = 0.0
        solverTime = np.zeros(n_steps)
        energy = np.zeros(n_steps)
        for i in range(n_steps):
            if i % 1000 == 0:
                print("time step : ", i)
            t += env._dt
            q_p_t, qdot_p_t, qddot_p_t = refTraj_obst.evaluate(t)
            q_g_t, qdot_g_t, qddot_g_t = refTraj_goal.evaluate(t)
            if e == 1:
                if i == 0:
                    print("run %i with no relative part" % e)
                qdot_g_t = np.zeros(2)
                qddot_g_t = np.zeros(2)
            t0 = time.time()
            q_t = ob[0:n]
            qdot_t = ob[n : 2 * n]
            action = planner.computeAction(
                q_t, qdot_t,
                q_p_t, qdot_p_t, qddot_p_t,
                q_g_t, qdot_g_t, qddot_g_t
            )
            en = exLag.evaluate(q_t, qdot_t)[2]
            # action = planner.computeAction(ob[0:n], ob[n:2*n])
            solverTime[i] = time.time() - t0
            energy[i] = en
            # env.render()
            ob, reward, done, info = env.step(action)
            q[i, :] = ob[0:n]
        qs.append(q)
        solverTimes.append(solverTime)
        energies.append(energy)
    ## Plotting the results
    res = {}
    res["qs"] = qs
    res["solverTimes"] = solverTimes
    res["obsts"] = obsts
    res["dt"] = env._dt
    res["x_d"] = x_goal
    res['energies'] = energies
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
    for i in range(2):
        solverAxs[0].plot(res['energies'][i])
    """
    solverPlot = SolverPlot(
        [res["solverTimes"][i] for i in sol_indices], 1, 1, axs=solverAxs
    )
    solverPlot.plot()
    """
    robotPlot.show()
