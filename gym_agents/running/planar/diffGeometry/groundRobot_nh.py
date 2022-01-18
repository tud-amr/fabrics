import gym
import groundRobots
import time
import casadi as ca
import numpy as np

from optFabrics.planner.nonHolonomicPlanner import (
    DefaultNonHolonomicPlanner,
)
from optFabrics.diffGeometry.diffMap import DifferentialMap, RelativeDifferentialMap
from optFabrics.diffGeometry.analyticSymbolicTrajectory import AnalyticSymbolicTrajectory
from optFabrics.diffGeometry.energy import Lagrangian
from optFabrics.planner.default_leaves import defaultAttractor
from optFabrics.planner.default_geometries import CollisionGeometry, GoalGeometry
from optFabrics.planner.default_maps import CollisionMap
from optFabrics.planner.default_energies import (
    CollisionLagrangian,
    ExecutionLagrangian,
)

from MotionPlanningEnv.sphereObstacle import SphereObstacle
from MotionPlanningEnv.dynamicSphereObstacle import DynamicSphereObstacle
from MotionPlanningGoal.staticSubGoal import StaticSubGoal


def pointMass(n_steps=5000):
    # setting up the problem
    nx = 3
    nu = 2
    staticObstDict1 = {'dim': 2, 'type': 'sphere', 'geometry': {'position': [0.0, -1.0], 'radius': 0.5}} 
    staticObstDict2 = {'dim': 2, 'type': 'sphere', 'geometry': {'position': [2.0, 1.5], 'radius': 0.5}} 
    obsts = [
        SphereObstacle(name="staticObst1", contentDict=staticObstDict1),
        SphereObstacle(name="staticObst2", contentDict=staticObstDict2),
    ]
    planner = DefaultNonHolonomicPlanner(nx, m_base=1.0)
    x, xdot, qdot = planner.vars()
    x_ee = x[0:2] + 0.8 *  ca.vertcat(ca.cos(x[2]), ca.sin(x[2]))
    # collision avoidance
    l_front = 0.800
    x_f = x[0:2] + ca.vertcat(l_front * ca.cos(x[2]), l_front * ca.sin(x[2]))
    x_col = ca.SX.sym("x_col", 1)
    xdot_col = ca.SX.sym("xdot_col", 1)
    lag_col = CollisionLagrangian(x_col, xdot_col)
    geo_col = CollisionGeometry(x_col, xdot_col)
    fks = [x_ee, x_f]
    r_body = 0.1
    for obst in obsts:
        for fk in fks:
            dm_col = CollisionMap(x, xdot, fk, obst.position(), obst.radius() + r_body)
            planner.addGeometry(dm_col, lag_col, geo_col)
    # forcing
    goalDict = {
        "m": 2,
        "w": 1.0,
        "prime": True,
        "indices": [0, 1],
        "parent_link": 0,
        "child_link": 3,
        "desired_position": [3.0, -0.7],
        "epsilon": 0.2,
        "type": "staticSubGoal",
    }
    goal = StaticSubGoal(name='goal', contentDict=goalDict)
    fk_ee = x_ee[0:2]
    x_psi = ca.SX.sym("x_psi", 2)
    dm_psi, lag_psi, _, x_psi, xdot_psi = defaultAttractor(x, xdot, goal.position(), fk_ee)
    geo_psi = GoalGeometry(x_psi, xdot_psi, k_psi=5)
    planner.addForcingGeometry(dm_psi, lag_psi, geo_psi)
    # finalize
    exLag = ExecutionLagrangian(x, xdot)
    x_ex = x
    xdot_ex = xdot
    exLag = ExecutionLagrangian(x_ex, xdot_ex)
    planner.setExecutionEnergy(exLag)
    exLag.concretize()
    # Speed control
    ex_factor = 1.0
    planner.setDefaultSpeedControl(
        x_psi, dm_psi, exLag, ex_factor,
        #r_b=0.5, b=[0.2, 15.0]
    )
    planner.concretize()
    # running the simulation
    env = gym.make("ground-robot-acc-v0", dt=0.010, render=True)
    ob = env.reset(pos=np.array([-5.0, 0.0, 0.0]), vel=np.array([1.0, 0.0]))
    for obst in obsts:
        env.addObstacle(obst)
    env.addGoal(goal)
    print("Starting episode")
    for i in range(n_steps):
        if i % 1000 == 0:
            print("time step : ", i)
        x = ob['x']
        xdot = ob['xdot']
        qdot = ob['vel']
        t0 = time.perf_counter()
        action = planner.computeAction(x, xdot, qdot)
        t1 = time.perf_counter()
        print(f"computation time in ms: {(t1 - t0)*1000}")
        ob, reward, done, info = env.step(action)


if __name__ == "__main__":
    n_steps = 70000
    pointMass(n_steps=n_steps)
