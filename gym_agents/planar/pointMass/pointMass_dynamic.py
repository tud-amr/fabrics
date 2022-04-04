import gym
import planarenvs.pointRobot
import time
import casadi as ca
import numpy as np

from fabrics.planner.fabricPlanner import DefaultFabricPlanner
from fabrics.planner.default_geometries import CollisionGeometry
from fabrics.planner.default_energies import (
    CollisionLagrangian,
    ExecutionLagrangian,
)
from fabrics.planner.default_maps import (
    CollisionMap,
)
from fabrics.planner.default_leaves import defaultDynamicAttractor
from fabrics.diffGeometry.diffMap import DifferentialMap, RelativeDifferentialMap
from fabrics.diffGeometry.analyticSymbolicTrajectory import AnalyticSymbolicTrajectory

from fabrics.helpers.variables import Variables

from MotionPlanningEnv.sphereObstacle import SphereObstacle
from MotionPlanningEnv.dynamicSphereObstacle import DynamicSphereObstacle
from MotionPlanningGoal.dynamicSubGoal import DynamicSubGoal


def pointMassDynamicGoal(n_steps=5000, render=True):
    env = gym.make("point-robot-acc-v0", dt=0.01, render=render)
    # obstacles
    obstTraj = ["0.5", "-3.0 * ca.sin(0.3 * t)"]
    dynamicObstDict = {'dim': 2, 'type': 'sphere', 'geometry': {'trajectory': obstTraj, 'radius': 1.0}} 
    staticObstDict = {'dim': 2, 'type': 'sphere', 'geometry': {'position': [2.0, 0.5], 'radius': 0.35}} 
    obsts = [
        DynamicSphereObstacle(name="dynamicObst", contentDict=dynamicObstDict),
        SphereObstacle(name="staticObst", contentDict=staticObstDict),
    ]
    # goal
    goalTraj = ["2.0 * ca.cos(0.3 * t)", "1.5 * ca.sin(0.3 * t)"]
    goalDict = {
        "m": 2,
        "w": 1.0,
        "prime": True,
        "indices": [0, 1],
        "parent_link": 0,
        "child_link": 3,
        "trajectory": goalTraj,
        "epsilon": 0.2,
        "type": "analyticSubGoal",
    }
    goal = DynamicSubGoal(name='goal', contentDict=goalDict)
    n = 2
    planner = DefaultFabricPlanner(n, m_base=1.0)
    var_q = planner.var()
    # collision avoidance
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    var_x = Variables(state_variables={'x': x, 'xdot': xdot})
    fks = [var_q.position_variable()]
    lag_col = CollisionLagrangian(var_x)
    geo_col = CollisionGeometry(var_x, exp=1)
    x_obst = ca.SX.sym("x_obst", 2)
    xdot_obst = ca.SX.sym("xdot_obst", 2)
    xddot_obst = ca.SX.sym("xddot_obst", 2)
    var_obst = Variables(parameters={'x_obst': x_obst, 'xdot_obst': xdot_obst, 'xddot_obst': xddot_obst})
    refTraj = AnalyticSymbolicTrajectory(ca.SX(np.identity(2)), 2, var=var_obst, traj=obstTraj)
    refTraj.concretize()
    for obst in obsts:
        x_col = ca.SX.sym("x_col", 2)
        xdot_col = ca.SX.sym("xdot_col", 2)
        var_col = Variables(state_variables={'x_col': x_col, 'xdot_col': xdot_col})
        x_rel = ca.SX.sym("x_rel", 2)
        xdot_rel = ca.SX.sym("xdot_rel", 2)
        var_rel = Variables(state_variables={'x_rel': x_rel, 'xdot_rel': xdot_rel})
        for fk in fks:
            if isinstance(obst, DynamicSphereObstacle):
                phi_n = ca.norm_2(x_rel) / obst.radius() - 1
                dm_n = DifferentialMap(phi_n, var=var_rel)
                dm_rel = RelativeDifferentialMap(var=var_q, refTraj=refTraj)
                dm_col = DifferentialMap(fk, var=var_q)
                planner.addGeometry(dm_col, lag_col.pull(dm_n).pull(dm_rel), geo_col.pull(dm_n).pull(dm_rel))
            elif isinstance(obst, SphereObstacle):
                dm_col = CollisionMap(var_q, fk, obst.position(), obst.radius())
                planner.addGeometry(dm_col, lag_col, geo_col)
    # forcing term
    x_goal = ca.SX.sym("x_obst", 2)
    xdot_goal = ca.SX.sym("xdot_obst", 2)
    xddot_goal = ca.SX.sym("xddot_obst", 2)
    var_goal = Variables(parameters={'x_goal': x_goal, 'xdot_goal': xdot_goal, 'xddot_goal': xddot_goal})
    goalSymbolicTraj = AnalyticSymbolicTrajectory(ca.SX(np.identity(2)), 2, var=var_goal, traj=goalTraj)
    goalSymbolicTraj.concretize()
    fk_ee = var_q.position_variable()
    dm_psi, lag_psi, geo_psi, var_psi = defaultDynamicAttractor(
        var_q, fk_ee, goalSymbolicTraj, k_psi=15.0
    )
    planner.addForcingGeometry(dm_psi, lag_psi, geo_psi, goalVelocity=goalSymbolicTraj.xdot())
    # execution energy
    exLag = ExecutionLagrangian(var_q)
    planner.setExecutionEnergy(exLag)
    # Speed control
    ex_factor = 1.0
    planner.setDefaultSpeedControl(var_psi.position_variable(), dm_psi, exLag, ex_factor, b=[2.0, 5.0])
    # planner.setConstantSpeedControl(beta=5.0)
    planner.concretize()
    # setup environment
    # running the simulation
    dynamicFabric = True
    x0 = np.array([2.3, -1.0])
    xdot0 = np.array([-1.0, -0.0])
    ob = env.reset(pos=x0, vel=xdot0)
    for obst in obsts:
        env.addObstacle(obst)
    env.addGoal(goal)
    print("Starting episode")
    for i in range(n_steps):
        if i % 1000 == 0:
            print("time step : ", i)
        q_p_t, qdot_p_t, qddot_p_t = refTraj.evaluate(env.t())
        q_g_t, qdot_g_t, qddot_g_t = goalSymbolicTraj.evaluate(env.t())
        if not dynamicFabric:
            qdot_g_t = np.zeros(2)
            qddot_g_t = np.zeros(2)
        action = planner.computeAction(
            q=ob['x'],
            qdot=ob['xdot'],
            x_obst=q_p_t,
            xdot_obst=qdot_p_t,
            xddot_obst=qddot_p_t,
            x_goal=q_g_t,
            xdot_goal=qdot_g_t,
            xddot_goal=qddot_g_t,
        )
        ob, reward, done, info = env.step(action)
    return {}


if __name__ == "__main__":
    n_steps = 4000
    pointMassDynamicGoal(n_steps=n_steps)
