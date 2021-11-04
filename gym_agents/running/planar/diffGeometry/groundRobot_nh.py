import gym
import groundRobots
import time
import casadi as ca
import numpy as np

from optFabrics.planner.nonHolonomicPlanner import NonHolonomicPlanner
from optFabrics.diffGeometry.diffMap import DifferentialMap, RelativeDifferentialMap
from optFabrics.diffGeometry.energy import FinslerStructure, Lagrangian
from optFabrics.diffGeometry.geometry import Geometry
from optFabrics.diffGeometry.referenceTrajectory import AnalyticTrajectory
from optFabrics.planner.default_leaves import defaultAttractor
from optFabrics.planner.default_geometries import CollisionGeometry, GoalGeometry
from optFabrics.planner.default_energies import GoalLagrangian
from optFabrics.planner.default_maps import GoalMap, CollisionMap
from optFabrics.planner.default_energies import CollisionLagrangian, ExecutionLagrangian, GoalLagrangian

from obstacle import Obstacle, DynamicObstacle
from robotPlot import RobotPlot
from solverPlot import SolverPlot


def pointMass(n_steps=5000):
    ## setting up the problem
    nx = 4
    nu = 3
    t = ca.SX.sym('t', 1)
    x_obst = ca.vertcat(0.5 - 0.1 * t, -3.0 + 1.2 * t)
    x_obst = ca.vertcat(3.0, 6.0 * ca.sin(0.1 * t))
    x_obst_fun = ca.Function("x_obst_fun", [t], [x_obst])
    refTraj_obst = AnalyticTrajectory(2, ca.SX(np.identity(2)), traj=x_obst, t=t, name="obst")
    refTraj_obst.concretize()
    r = 1.0
    obsts = [
                Obstacle(np.array([0.0, 0.4]), 1.2),
                DynamicObstacle(x_obst_fun, r),
            ]
    x = ca.SX.sym("x", nx)
    xdot = ca.SX.sym("xdot", nx)
    # actions forward velocity
    qdot = ca.SX.sym("qdot", nu)
    M_base = np.identity(nx) * 1.0
    M_base[3, 3] = 3
    l = 0.5 * ca.dot(xdot, ca.mtimes(M_base, xdot))
    l_base = Lagrangian(l, x=x, xdot=xdot)
    h_base = ca.SX(np.zeros(nx))
    geo_base = Geometry(h=h_base, var=[x, xdot])
    J_nh = ca.SX(
        np.array([
            [ca.cos(x[2]), 0, 0],
            [ca.sin(x[2]), 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
    )
    f_extra = qdot[0] * qdot[1] * ca.vertcat(-ca.sin(x[2]), ca.cos(x[2]), 0, 0)
    #f_extra = np.zeros(4)
    planner = NonHolonomicPlanner(geo_base, l_base, J_nh, qdot, f_extra)
    x_ee = x[0:2] + 0.2 * ca.vertcat(ca.cos(x[2]), ca.sin(x[2])) + 1.0 * ca.vertcat(ca.cos(x[2] + x[3]), ca.sin(x[2] + x[3]))
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
    x_corners = [x_f, x_b, x_r, x_l, x_ee]
    x_corners = [x_f, x_ee, x_b, x_r, x_l]
    x_corners = [x_ee, x[0:2]]
    r_corners = [0.2, 1.0]
    lams = np.array([5.5, 4.0, 1.0, 1.0, 1.5] )
    x_col1d = ca.SX.sym("x_col1d", 1)
    xdot_col1d = ca.SX.sym("xdot_col1d", 1)
    lag_col = CollisionLagrangian(x_col1d, xdot_col1d)
    for obst in obsts:
        x_col = ca.SX.sym("x_col", 2)
        xdot_col = ca.SX.sym("xdot_col", 2)
        x_rel = ca.SX.sym("x_rel", 2)
        xdot_rel = ca.SX.sym("xdot_rel", 2)
        for i, x_corner in enumerate(x_corners):
            r_corner = r_corners[i]
            geo_col = CollisionGeometry(x_col1d, xdot_col1d, lam=lams[i], exp=3)
            if isinstance(obst, DynamicObstacle):
                phi_n = ca.norm_2(x_rel)/ (obst.r() + r_corner) - 1
                dm_n = DifferentialMap(phi_n, q=x_rel, qdot=xdot_rel)
                dm_rel = RelativeDifferentialMap(q=x_col, qdot=xdot_col, refTraj=refTraj_obst)
                dm_col = DifferentialMap(x_corner, q=x, qdot=xdot)
                planner.addGeometry(dm_col, lag_col.pull(dm_n).pull(dm_rel), geo_col.pull(dm_n).pull(dm_rel))
            elif isinstance(obst, Obstacle):
                dm_col = CollisionMap(x, xdot, x_corner, obst.x(), obst.r())
                planner.addGeometry(dm_col, lag_col, geo_col)
    """
    for obst in obsts:
        for i, x_corner in enumerate(x_corners):
            if isinstance(obst, DynamicObstacle):
                dm_col = VariableCollisionMap(x, xdot, x_corner, obst.r(), q_p, qdot_p, qddot_p)
            elif isinstance(obst, Obstacle):
                dm_col = CollisionMap(x, xdot, x_corner, obst.x(), obst.r())
            le = lams[i] * 1/(x_col**1) * s * xdot_col**2
            lag_col = Lagrangian(le, x=x_col, xdot=xdot_col)
            planner.addGeometry(dm_col, lag_col, geo)
    """
    # forcing
    x_d = np.array([6.0, 3.0])
    x_psi = ca.SX.sym("x_psi", 2)
    xdot_psi = ca.SX.sym("xdot_psi", 2)
    phi_psi = x_ee - x_d
    dm_psi = DifferentialMap(phi_psi, q=x, qdot=xdot)
    dm_psi.concretize()
    lag_psi = GoalLagrangian(x_psi, xdot_psi, k_psi=0.5)
    geo_psi = GoalGeometry(x_psi, xdot_psi)
    geo_psi.concretize()
    planner.addForcingGeometry(dm_psi, lag_psi, geo_psi)
    # finalize
    exLag = ExecutionLagrangian(x, xdot)
    l_ex = 0.5 * ca.dot(xdot[0:2], xdot[0:2]) + xdot[3]**2
    exLag = Lagrangian(l_ex, x=x, xdot=xdot)
    # exLag.concretize()
    planner.setExecutionEnergy(exLag)
    # Speed control
    ex_factor = 0.1
    #planner.setConstantSpeedControl(beta=0.1)
    planner.setDefaultSpeedControl(x_psi, dm_psi, exLag, ex_factor, r_b=2.0)
    planner.concretize()
    # setup environment
    qs = []
    solverTimes = []
    # running the simulation
    env = gym.make('ground-robot-diffdrive-arm-acc-v0', dt=0.010)
    ob = env.reset(np.array([-5.0, 1.0, 0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0]))
    print("Starting episode")
    q = np.zeros((n_steps, nx))
    t = 0.0
    solverTime = np.zeros(n_steps)
    for i in range(n_steps):
        if i % 1000 == 0:
            print('time step : ', i)
        t += env._dt
        t0 = time.time()
        x = ob[0:nx]
        xdot = ob[nx:2*nx]
        qdot = np.concatenate((ob[-2:], xdot[-1:]))
        q_p_t, qdot_p_t, qddot_p_t = refTraj_obst.evaluate(t)
        # action = planner.computeAction(x, xdot, qdot)
        action = planner.computeAction(x, xdot, q_p_t, qdot_p_t, qddot_p_t, qdot)
        solverTime[i] = time.time() - t0
        # env.render()
        ob, reward, done, info = env.step(action)
        q[i, 0:4] = ob[0:4]
    qs.append(q)
    solverTimes.append(solverTime)
    ## Plotting the results
    res = {}
    res['qs'] = qs
    res['solverTimes'] = solverTimes
    res['obsts'] = obsts
    res['dt'] = env._dt
    res['x_ee_fun'] = x_ee_fun
    return res

if __name__ == "__main__":
    n_steps = 6000
    res = pointMass(n_steps=n_steps)
    fk_fun = res['x_ee_fun']
    sol_indices = [0]
    robotPlot = RobotPlot([res['qs'][i] for i in sol_indices], fk_fun, 2, types=[4])
    lim = 8
    robotPlot.initFig(2, 1, lims=[(-lim, lim), (-lim, lim)])
    robotPlot.addObstacle([0], res['obsts'])
    robotPlot.plot()
    robotPlot.makeAnimation(n_steps)
    solverAxs = robotPlot.getAxs([1])
    solverPlot = SolverPlot([res['solverTimes'][i] for i in sol_indices], 1, 1, axs=solverAxs)
    solverPlot.plot()
    robotPlot.show()
