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

from fabrics.helpers.variables import Variables

from fabrics.diffGeometry.diffMap import DifferentialMap, RelativeDifferentialMap
from fabrics.diffGeometry.analyticSymbolicTrajectory import AnalyticSymbolicTrajectory

from MotionPlanningEnv.sphereObstacle import SphereObstacle
from MotionPlanningEnv.dynamicSphereObstacle import DynamicSphereObstacle
from MotionPlanningGoal.dynamicSubGoal import DynamicSubGoal


def pointMassDynamic(n_steps=5000):
    ## setting up the problem
    n = 2
    env = gym.make('point-robot-acc-v0', dt=0.005)
    obst1_traj = ["-1.0 - 3.5 * ca.cos(1.0 * t)", "-0.5 - 3.5 * ca.sin(1.0 * t)"]
    x_obst1 = ca.SX.sym("x_obst1", 2)
    xdot_obst1 = ca.SX.sym("xdot_obst1", 2)
    xddot_obst1 = ca.SX.sym("xddot_obst1", 2)
    var_obst1 = Variables(parameters={'x_obst1': x_obst1, 'xdot_obst1': xdot_obst1, 'xddot_obst1': xddot_obst1})
    refTraj_obst1 = AnalyticSymbolicTrajectory(ca.SX(np.identity(2)), 2, var=var_obst1, traj=obst1_traj)
    refTraj_obst1.concretize()
    dynamicObstDict1 = {'dim': 2, 'type': 'sphere', 'geometry': {'trajectory': obst1_traj, 'radius': 1.0}} 
    obst2_traj = ["0.5 + 2.0 * ca.cos(0.5 * t)", "-1.0 - 2.0 * ca.sin(0.5 * t)"]
    x_obst2 = ca.SX.sym("x_obst2", 2)
    xdot_obst2 = ca.SX.sym("xdot_obst2", 2)
    xddot_obst2 = ca.SX.sym("xddot_obst2", 2)
    var_obst2 = Variables(parameters={'x_obst2': x_obst2, 'xdot_obst2': xdot_obst2, 'xddot_obst2': xddot_obst2})
    refTraj_obst2 = AnalyticSymbolicTrajectory(ca.SX(np.identity(2)), 2, var=var_obst2, traj=obst2_traj)
    refTraj_obst2.concretize()
    refTrajs = [refTraj_obst1, refTraj_obst2]
    dynamicObstDict2 = {'dim': 2, 'type': 'sphere', 'geometry': {'trajectory': obst2_traj, 'radius': 0.5}} 
    obsts = [
        DynamicSphereObstacle(name="dynamicObst1", contentDict=dynamicObstDict1),
        DynamicSphereObstacle(name="dynamicObst2", contentDict=dynamicObstDict2),
    ]
    planner = DefaultFabricPlanner(n, m_base=1.0)
    var_q = planner.var()
    fks = [var_q.position_variable()]
    # collision avoidance
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    var_x = Variables(state_variables={'x': x, 'xdot': xdot})
    lag_col = CollisionLagrangian(var_x)
    geo_col = CollisionGeometry(var_x, exp=2, lam=0.1)
    for i, obst in enumerate(obsts):
        x_col = ca.SX.sym("x_col", 2)
        xdot_col = ca.SX.sym("xdot_col", 2)
        var_col = Variables(state_variables={'x_col': x_col, 'xdot_col': xdot_col})
        x_rel = ca.SX.sym("x_rel", 2)
        xdot_rel = ca.SX.sym("xdot_rel", 2)
        var_rel = Variables(state_variables={'x_rel': x_rel, 'xdot_rel': xdot_rel})
        for fk in fks:
            if isinstance(obst, DynamicSphereObstacle):
                refTraj = refTrajs[i]
                phi_n = ca.norm_2(x_rel) / obst.radius() - 1
                dm_n = DifferentialMap(phi_n, var=var_rel)
                dm_rel = RelativeDifferentialMap(var=var_col, refTraj=refTraj)
                dm_col = DifferentialMap(fk, var=var_q)
                planner.addGeometry(dm_col, lag_col.pull(dm_n).pull(dm_rel), geo_col.pull(dm_n).pull(dm_rel))
            elif isinstance(obst, SphereObstacle):
                dm_col = CollisionMap(var_q, fk, obst.position(), obst.radius())
                planner.addGeometry(dm_col, lag_col, geo_col)
    # joint limit avoidance
    lag_lim = CollisionLagrangian(var_x)
    geo_lim = LimitGeometry(var_x, lam=15.00, exp=2)
    dm_lim_upper = UpperLimitMap(var_q, 10.0, 0)
    planner.addGeometry(dm_lim_upper, lag_lim, geo_lim)
    dm_lim_upper = UpperLimitMap(var_q, 7.0, 1)
    planner.addGeometry(dm_lim_upper, lag_lim, geo_lim)
    dm_lim_lower = LowerLimitMap(var_q, -10.0, 0)
    planner.addGeometry(dm_lim_lower, lag_lim, geo_lim)
    dm_lim_lower = LowerLimitMap(var_q, -7.0, 1)
    planner.addGeometry(dm_lim_lower, lag_lim, geo_lim)
    # forcing term
    q_d = np.array([0, 0])
    dm_psi, lag_psi, _, var_psi  = defaultAttractor(var_q, q_d, fk)
    geo_psi = GoalGeometry(var_psi, k_psi=3.0)
    planner.addForcingGeometry(dm_psi, lag_psi, geo_psi)
    # execution energy
    exLag = ExecutionLagrangian(var_q)
    #planner.setExecutionEnergy(exLag)
    # Speed control
    ex_factor = 1.0
    #planner.setDefaultSpeedControl(var_psi.position_variable(), dm_psi, exLag, ex_factor)
    planner.setConstantSpeedControl(beta=0.1)
    planner.concretize()
    # setup environment
    qs = []
    solverTimes = []
    # running the simulation
    env = gym.make('point-robot-acc-v0', dt=0.01, render=True)
    ob = env.reset(pos=np.array([4, 4]))
    for obst in obsts:
        env.add_obstacle(obst)
        env.resetLimits(pos={'low': np.array([-10, -7]), 'high': np.array([10, 7])})
    solverTime = np.zeros(n_steps)
    for i in range(n_steps):
        """
        if i % 100 == 0:
            print('time step : ', i)
        """
        q_p_t, qdot_p_t, qddot_p_t = refTraj_obst1.evaluate(env.t())
        q2_p_t, q2dot_p_t, q2ddot_p_t = refTraj_obst2.evaluate(env.t())
        #qdot_p_t = np.zeros(2)
        #q2dot_p_t = np.zeros(2)
        #qddot_p_t = np.zeros(2)
        #q2ddot_p_t = np.zeros(2)
        action = planner.computeAction(
            q=ob['x'],
            qdot=ob['xdot'],
            x_obst1=q_p_t,
            xdot_obst1=qdot_p_t,
            xddot_obst1=qddot_p_t,
            x_obst2=q2_p_t,
            xdot_obst2=q2dot_p_t,
            xddot_obst2=q2ddot_p_t
        )
        #print(en_ex)
        ob, reward, done, info = env.step(action)
    ## Plotting the results
    res = {}
    return res

if __name__ == "__main__":
    n_steps = 10000
    res = pointMassDynamic(n_steps=n_steps)
