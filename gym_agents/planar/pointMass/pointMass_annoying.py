import gym
import planarenvs.point_robot
import time
import casadi as ca
import numpy as np

from fabrics.planner.fabricPlanner import DefaultFabricPlanner
from fabrics.planner.default_geometries import CollisionGeometry, GoalGeometry
from fabrics.planner.default_geometries import CollisionGeometry, LimitGeometry, GoalGeometry
from fabrics.planner.default_energies import CollisionLagrangian, ExecutionLagrangian
from fabrics.planner.default_maps import CollisionMap, UpperLimitMap, LowerLimitMap
from fabrics.planner.default_leaves import defaultAttractor

from fabrics.diffGeometry.diffMap import DifferentialMap, RelativeDifferentialMap
from fabrics.diffGeometry.analyticSymbolicTrajectory import AnalyticSymbolicTrajectory

from MotionPlanningEnv.sphereObstacle import SphereObstacle
from MotionPlanningEnv.dynamicSphereObstacle import DynamicSphereObstacle
from MotionPlanningGoal.dynamicSubGoal import DynamicSubGoal


def pointMassDynamic(n_steps=5000):
    ## setting up the problem
    n = 2
    env = gym.make('point-robot-acc-v0', dt=0.005)
    obst1_traj = ["-1.0 - 3.5 * ca.cos(0.4 * t)", "-0.5 - 3.5 * ca.sin(0.4 * t)"]
    refTraj_obst1 = AnalyticSymbolicTrajectory(ca.SX(np.identity(2)), 2, traj=obst1_traj)
    refTraj_obst1.concretize()
    dynamicObstDict1 = {'dim': 2, 'type': 'sphere', 'geometry': {'trajectory': obst1_traj, 'radius': 1.0}} 
    obst2_traj = ["0.5 + 2.0 * ca.cos(0.5 * t)", "-1.0 - 2.0 * ca.sin(0.5 * t)"]
    refTraj_obst2 = AnalyticSymbolicTrajectory(ca.SX(np.identity(2)), 2, traj=obst2_traj)
    refTraj_obst2.concretize()
    refTrajs = [refTraj_obst1, refTraj_obst2]
    dynamicObstDict2 = {'dim': 2, 'type': 'sphere', 'geometry': {'trajectory': obst2_traj, 'radius': 0.5}} 
    obsts = [
        DynamicSphereObstacle(name="dynamicObst1", contentDict=dynamicObstDict1),
        DynamicSphereObstacle(name="dynamicObst2", contentDict=dynamicObstDict2),
    ]
    planner = DefaultFabricPlanner(n, m_base=0.1)
    q, qdot = planner.var()
    fks = [q]
    # collision avoidance
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    lag_col = CollisionLagrangian(x, xdot)
    geo_col = CollisionGeometry(x, xdot, exp=2, lam=5)
    for i, obst in enumerate(obsts):
        x_col = ca.SX.sym("x_col", 2)
        xdot_col = ca.SX.sym("xdot_col", 2)
        x_rel = ca.SX.sym("x_rel", 2)
        xdot_rel = ca.SX.sym("xdot_rel", 2)
        for fk in fks:
            if isinstance(obst, DynamicSphereObstacle):
                refTraj = refTrajs[i]
                phi_n = ca.norm_2(x_rel) / obst.radius() - 1
                dm_n = DifferentialMap(phi_n, q=x_rel, qdot=xdot_rel)
                dm_rel = RelativeDifferentialMap(q=x_col, qdot=xdot_col, refTraj=refTraj)
                dm_col = DifferentialMap(fk, q=q, qdot=qdot)
                planner.addGeometry(dm_col, lag_col.pull(dm_n).pull(dm_rel), geo_col.pull(dm_n).pull(dm_rel))
            elif isinstance(obst, SphereObstacle):
                dm_col = CollisionMap(q, qdot, fk, obst.position(), obst.radius())
                planner.addGeometry(dm_col, lag_col, geo_col)
    # joint limit avoidance
    lag_lim = CollisionLagrangian(x, xdot)
    geo_lim = LimitGeometry(x, xdot, lam=1.00, exp=2)
    dm_lim_upper = UpperLimitMap(q, qdot, 4.0, 0)
    planner.addGeometry(dm_lim_upper, lag_lim, geo_lim)
    dm_lim_upper = UpperLimitMap(q, qdot, 4.0, 1)
    planner.addGeometry(dm_lim_upper, lag_lim, geo_lim)
    dm_lim_lower = LowerLimitMap(q, qdot, -4.0, 0)
    planner.addGeometry(dm_lim_lower, lag_lim, geo_lim)
    dm_lim_lower = LowerLimitMap(q, qdot, -4.0, 1)
    planner.addGeometry(dm_lim_lower, lag_lim, geo_lim)
    # forcing term
    q_d = np.array([0.0, 0.0])
    dm_psi, lag_psi, _, x_psi, xdot_psi  = defaultAttractor(q, qdot, q_d, fk)
    geo_psi = GoalGeometry(x_psi, xdot_psi, k_psi=2.0)
    planner.addForcingGeometry(dm_psi, lag_psi, geo_psi)
    # execution energy
    exLag = ExecutionLagrangian(q, qdot)
    planner.setExecutionEnergy(exLag)
    # Speed control
    ex_factor = 1.0
    planner.setDefaultSpeedControl(x_psi, dm_psi, exLag, ex_factor)
    planner.concretize()
    # setup environment
    qs = []
    solverTimes = []
    x0s = [np.array([2.3, -1.0 + i * 0.2]) for i in range(1)]
    xdot0s = [np.array([np.cos(i*np.pi/5), np.sin(i*np.pi/5)]) for i in range(10)]
    x0s = [np.array([2.0, 1.0])]
    xdot0s = [np.ones(2) * -0.1]
    # running the simulation
    for xdot0 in xdot0s:
        for x0 in x0s:
            env = gym.make('point-robot-acc-v0', dt=0.01, render=True)
            ob = env.reset(pos=x0, vel=xdot0)
            for obst in obsts:
                env.add_obstacle(obst)
            print("Starting episode")
            q = np.zeros((n_steps, n))
            t = 0.0
            solverTime = np.zeros(n_steps)
            for i in range(n_steps):
                """
                if i % 100 == 0:
                    print('time step : ', i)
                """
                t0 = time.time()
                q_p_t, qdot_p_t, qddot_p_t = refTraj_obst1.evaluate(t)
                q2_p_t, q2dot_p_t, q2ddot_p_t = refTraj_obst2.evaluate(t)
                #qdot_p_t = np.zeros(2)
                #q2dot_p_t = np.zeros(2)
                #qddot_p_t = np.zeros(2)
                #q2ddot_p_t = np.zeros(2)
                action = planner.computeAction(
                    ob['x'], ob['xdot'],
                    q_p_t, qdot_p_t, qddot_p_t,
                    q2_p_t ,q2dot_p_t, q2ddot_p_t
                )
                #print(en_ex)
                solverTime[i] = time.time() - t0
                # env.render()
                ob, reward, done, info = env.step(action)
                q[i, :] = ob['x']
            qs.append(q)
            solverTimes.append(solverTime)
    ## Plotting the results
    res = {}
    res['qs'] = qs
    res['solverTimes'] = solverTimes
    res['obsts'] = obsts
    res['dt'] = env.dt()
    return res

if __name__ == "__main__":
    n_steps = 10000
    res = pointMassDynamic(n_steps=n_steps)
