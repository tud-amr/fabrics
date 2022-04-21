import gym
import planarenvs.point_robot
import time
import casadi as ca
import numpy as np

from fabrics.planner.fabricPlanner import DefaultFabricPlanner
from fabrics.defaults.default_geometries import CollisionGeometry
from fabrics.defaults.default_energies import (
    CollisionLagrangian,
    ExecutionLagrangian,
)
from fabrics.defaults.default_maps import (
    CollisionMap,
)
from fabrics.defaults.default_leaves import defaultDynamicAttractor, defaultAttractor
from fabrics.diffGeometry.diffMap import DifferentialMap, RelativeDifferentialMap
from fabrics.diffGeometry.analyticSymbolicTrajectory import AnalyticSymbolicTrajectory

from fabrics.helpers.variables import Variables

from MotionPlanningEnv.sphereObstacle import SphereObstacle
from MotionPlanningEnv.dynamicSphereObstacle import DynamicSphereObstacle
from MotionPlanningGoal.staticSubGoal import StaticSubGoal


def pointMassDynamicAnnoying(n_steps=5000, render=True):
    env = gym.make("point-robot-acc-v0", dt=0.01, render=render)
    # obstacles
    obstTraj = ["0.5", "6.0 * ca.cos(0.2 * t)"]
    dynamicObstDict = {'dim': 2, 'type': 'sphere', 'geometry': {'trajectory': obstTraj, 'radius': 1.0}} 
    staticObstDict = {'dim': 2, 'type': 'sphere', 'geometry': {'position': [2.0, 0.5], 'radius': 0.35}} 
    obsts = [
        DynamicSphereObstacle(name="dynamicObst", contentDict=dynamicObstDict),
        SphereObstacle(name="staticObst", contentDict=staticObstDict),
    ]
    # goal
    goalDict = {
        "m": 2,
        "w": 1.0,
        "prime": True,
        "indices": [0, 1],
        "parent_link": 0,
        "child_link": 3,
        "desired_position": [0.0, -3.0], 
        "epsilon": 0.2,
        "type": "analyticSubGoal",
    }
    goal = StaticSubGoal(name='goal', contentDict=goalDict)
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
    x_ref = ca.SX.sym("x_ref", 2)
    xdot_ref = ca.SX.sym("xdot_ref", 2)
    xddot_ref = ca.SX.sym("xddot_ref", 2)
    var_ref = Variables(parameters={'x_ref': x_ref, 'xdot_ref': xdot_ref, 'xddot_ref': xddot_ref})
    refTraj = AnalyticSymbolicTrajectory(ca.SX(np.identity(2)), 2, var=var_ref, traj=obstTraj)
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
                dm_rel = RelativeDifferentialMap(var=var_col, refTraj=refTraj)
                dm_fk = DifferentialMap(fk, var=var_q)
                planner.addGeometry(dm_fk, lag_col.pull(dm_n).pull(dm_rel), geo_col.pull(dm_n).pull(dm_rel))
                #dm_col = CollisionMap(var_q, fk, refTraj.x(), obst.radius())
                #planner.addGeometry(dm_col, lag_col, geo_col)
            elif isinstance(obst, SphereObstacle):
                dm_col = CollisionMap(var_q, fk, obst.position(), obst.radius())
                planner.addGeometry(dm_col, lag_col, geo_col)
    # forcing term
    fk_ee = var_q.position_variable()
    dm_psi, lag_psi, geo_psi, var_psi = defaultAttractor(
        var_q, goal.position(), fk_ee, k_psi=1.0
    )
    planner.addForcingGeometry(dm_psi, lag_psi, geo_psi)
    # execution energy
    exLag = ExecutionLagrangian(var_q)
    planner.setExecutionEnergy(exLag)
    # Speed control
    ex_factor = 1.0
    planner.setDefaultSpeedControl(var_psi.position_variable(), dm_psi, exLag, ex_factor, b=[0.01, 5.0])
    # planner.setConstantSpeedControl(beta=5.0)
    planner.concretize()
    # setup environment
    # running the simulation
    dynamicFabric = False
    x0 = np.array([0.3, 4.0])
    xdot0 = np.array([-1.0, -0.0])
    ob = env.reset(pos=x0, vel=xdot0)
    for obst in obsts:
        env.add_obstacle(obst)
    env.add_goal(goal)
    print("Starting episode")
    for i in range(n_steps):
        if i % 1000 == 0:
            print("time step : ", i)
        q_p_t, qdot_p_t, qddot_p_t = refTraj.evaluate(env.t())
        if not dynamicFabric:
            qdot_g_t = np.zeros(2)
            qddot_g_t = np.zeros(2)
        action = planner.computeAction(
            q=ob['x'],
            qdot=ob['xdot'],
            x_ref=q_p_t,
            xdot_ref=qdot_p_t,
            xddot_ref=qddot_p_t,
        )
        ob, reward, done, info = env.step(action)
    return {}


if __name__ == "__main__":
    n_steps = 4000
    pointMassDynamicAnnoying(n_steps=n_steps)
