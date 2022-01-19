import gym
import pointRobot
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
from fabrics.planner.default_leaves import defaultDynamicAttractor, defaultAttractor
from fabrics.diffGeometry.diffMap import DifferentialMap, RelativeDifferentialMap
from fabrics.diffGeometry.analyticSymbolicTrajectory import AnalyticSymbolicTrajectory

from MotionPlanningEnv.sphereObstacle import SphereObstacle
from MotionPlanningEnv.dynamicSphereObstacle import DynamicSphereObstacle
from MotionPlanningGoal.staticSubGoal import StaticSubGoal

def pointMassDynamicGoal(n_steps=5000):
    env = gym.make("point-robot-acc-v0", dt=0.01, render=True)
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
    q, qdot = planner.var()
    # collision avoidance
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    fks = [q]
    lag_col = CollisionLagrangian(x, xdot)
    geo_col = CollisionGeometry(x, xdot, exp=1)
    refTraj = AnalyticSymbolicTrajectory(ca.SX(np.identity(2)), 2, traj=obstTraj)
    refTraj.concretize()
    for obst in obsts:
        x_col = ca.SX.sym("x_col", 2)
        xdot_col = ca.SX.sym("xdot_col", 2)
        x_rel = ca.SX.sym("x_rel", 2)
        xdot_rel = ca.SX.sym("xdot_rel", 2)
        for fk in fks:
            if isinstance(obst, DynamicSphereObstacle):
                phi_n = ca.norm_2(x_rel) / obst.radius() - 1
                dm_n = DifferentialMap(phi_n, q=x_rel, qdot=xdot_rel)
                dm_rel = RelativeDifferentialMap(q=x_col, qdot=xdot_col, refTraj=refTraj)
                dm_fk = DifferentialMap(fk, q=q, qdot=qdot)
                planner.addGeometry(dm_fk, lag_col.pull(dm_n).pull(dm_rel), geo_col.pull(dm_n).pull(dm_rel))
                #dm_col = CollisionMap(q, qdot, fk, refTraj.x(), obst.radius())
                #planner.addGeometry(dm_col, lag_col, geo_col)
            elif isinstance(obst, SphereObstacle):
                dm_col = CollisionMap(q, qdot, fk, obst.position(), obst.radius())
                planner.addGeometry(dm_col, lag_col, geo_col)
    # forcing term
    fk_ee = q
    dm_psi, lag_psi, geo_psi, x_psi, xdot_psi = defaultAttractor(
        q, qdot, goal.position(), fk_ee, k_psi=1.0
    )
    planner.addForcingGeometry(dm_psi, lag_psi, geo_psi)
    # execution energy
    exLag = ExecutionLagrangian(q, qdot)
    planner.setExecutionEnergy(exLag)
    # Speed control
    ex_factor = 1.0
    planner.setDefaultSpeedControl(x_psi, dm_psi, exLag, ex_factor, b=[0.01, 5.0])
    # planner.setConstantSpeedControl(beta=5.0)
    planner.concretize()
    # setup environment
    # running the simulation
    dynamicFabric = False
    x0 = np.array([0.3, 4.0])
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
        if not dynamicFabric:
            qdot_g_t = np.zeros(2)
            qddot_g_t = np.zeros(2)
        action = planner.computeAction(
            ob['x'],
            ob['xdot'],
            q_p_t,
            qdot_p_t,
            qddot_p_t,
        )
        ob, reward, done, info = env.step(action)


if __name__ == "__main__":
    n_steps = 4000
    pointMassDynamicGoal(n_steps=n_steps)
