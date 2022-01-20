import gym
import planarenvs.nLinkReacher
import casadi as ca
import numpy as np
import time

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

from MotionPlanningEnv.sphereObstacle import SphereObstacle
from MotionPlanningEnv.dynamicSphereObstacle import DynamicSphereObstacle
from MotionPlanningGoal.dynamicSubGoal import DynamicSubGoal

from forwardkinematics.planarFks.planarArmFk import PlanarArmFk


def nlinkDynamicGoal(n=3, n_steps=5000, render=True):
    env = gym.make("nLink-reacher-acc-v0", n=n, dt=0.01, render=render)
    obstTraj = ["5 * ca.sin(0.3 * t)", "-3.0"]
    dynamicObstDict = {'dim': 2, 'type': 'sphere', 'geometry': {'trajectory': obstTraj, 'radius': 1.0}} 
    staticObstDict = {'dim': 2, 'type': 'sphere', 'geometry': {'position': [0.0, 3.0], 'radius': 1.0}} 
    obsts = [
        DynamicSphereObstacle(name="dynamicObst", contentDict=dynamicObstDict),
        SphereObstacle(name="staticObst", contentDict=staticObstDict),
    ]
    # goal
    goalTraj = ["1.5 + 0.7 * ca.sin(1.0 * t)", "-1 + 1 * ca.cos(1.0 * t)"]
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
    planner = DefaultFabricPlanner(n, m_base=1.0)
    q, qdot = planner.var()
    planarArmFk = PlanarArmFk(n)
    # collision avoidance
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    lag_col = CollisionLagrangian(x, xdot)
    geo_col = CollisionGeometry(x, xdot, lam=20, exp=2)
    refTraj = AnalyticSymbolicTrajectory(ca.SX(np.identity(2)), 2, traj=obstTraj)
    refTraj.concretize()
    for obst in obsts:
        x_col = ca.SX.sym("x_col", 2)
        xdot_col = ca.SX.sym("xdot_col", 2)
        x_rel = ca.SX.sym("x_rel", 2)
        xdot_rel = ca.SX.sym("xdot_rel", 2)
        for i in range(1, n+1):
            fk = planarArmFk.fk(q, i, positionOnly=True)
            if isinstance(obst, DynamicSphereObstacle):
                phi_n = ca.norm_2(x_rel)/ obst.radius() - 1
                dm_n = DifferentialMap(phi_n, q=x_rel, qdot=xdot_rel)
                dm_rel = RelativeDifferentialMap(q=x_col, qdot=xdot_col, refTraj=refTraj)
                dm_col = DifferentialMap(fk, q=q, qdot=qdot)
                planner.addGeometry(dm_col, lag_col.pull(dm_n).pull(dm_rel), geo_col.pull(dm_n).pull(dm_rel))
            elif isinstance(obst, SphereObstacle):
                dm_col = CollisionMap(q, qdot, fk, obst.position(), obst.radius())
                planner.addGeometry(dm_col, lag_col, geo_col)
    # forcing term
    goalSymbolicTraj = AnalyticSymbolicTrajectory(ca.SX(np.identity(2)), 2, traj=goalTraj)
    goalSymbolicTraj.concretize()
    fk_ee = planarArmFk.fk(q, n, positionOnly=True)
    dm_psi, lag_psi, geo_psi, x_psi, xdot_psi = defaultDynamicAttractor(
        q, qdot, fk_ee, goalSymbolicTraj, k_psi=10.0
    )
    planner.addForcingGeometry(dm_psi, lag_psi, geo_psi, goalVelocity=goalSymbolicTraj.xdot())
    # execution energy
    exLag = ExecutionLagrangian(q, qdot)
    exLag.concretize()
    planner.setExecutionEnergy(exLag)
    # Speed control
    ex_factor = 1.0
    planner.setDefaultSpeedControl(x_psi, dm_psi, exLag, ex_factor, b=[0.05, 8.0])
    planner.concretize()
    dynamicFabrics = False
    ob = env.reset()
    for obst in obsts:
        env.addObstacle(obst)
    env.addGoal(goal)
    print("Starting episode")
    for i in range(n_steps):
        if i % 1000 == 0:
            print("time step : ", i)
        q_p_t, qdot_p_t, qddot_p_t = refTraj.evaluate(env.t())
        q_g_t, qdot_g_t, qddot_g_t = goalSymbolicTraj.evaluate(env.t())
        if not dynamicFabrics:
            qdot_g_t = np.zeros(2)
            qddot_g_t = np.zeros(2)
        q_t = ob['x']
        qdot_t = ob['xdot']
        t0 = time.perf_counter()
        action = planner.computeAction(
            q_t, qdot_t,
            q_p_t, qdot_p_t, qddot_p_t,
            q_g_t, qdot_g_t, qddot_g_t
        )
        t1 = time.perf_counter()
        print(f"computational time in ms: {(t1 - t0)*1000}")
        ob, reward, done, info = env.step(action)


if __name__ == "__main__":
    n_steps = 5000
    n = 3
    nlinkDynamicGoal(n=n, n_steps=n_steps)
