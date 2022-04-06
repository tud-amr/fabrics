import gym
import planarenvs.ground_robots
import time
import casadi as ca
import numpy as np

from fabrics.planner.nonHolonomicPlanner import (
    DefaultNonHolonomicPlanner,
)
from fabrics.diffGeometry.diffMap import DifferentialMap, RelativeDifferentialMap
from fabrics.diffGeometry.analyticSymbolicTrajectory import AnalyticSymbolicTrajectory
from fabrics.diffGeometry.energy import Lagrangian
from fabrics.planner.default_leaves import defaultAttractor
from fabrics.planner.default_geometries import CollisionGeometry, GoalGeometry
from fabrics.planner.default_maps import CollisionMap
from fabrics.planner.default_energies import (
    CollisionLagrangian,
    ExecutionLagrangian,
)

from fabrics.helpers.variables import Variables

from MotionPlanningEnv.sphereObstacle import SphereObstacle
from MotionPlanningEnv.dynamicSphereObstacle import DynamicSphereObstacle
from MotionPlanningGoal.staticSubGoal import StaticSubGoal


def ground_robot(n_steps=5000):
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
    var_q, qudot = planner.vars()
    q = var_q.position_variable()
    x_ee = q[0:2] + 0.8 *  ca.vertcat(ca.cos(q[2]), ca.sin(q[2]))
    # collision avoidance
    l_front = 0.800
    x_f = q[0:2] + ca.vertcat(l_front * ca.cos(q[2]), l_front * ca.sin(q[2]))
    x_col = ca.SX.sym("x_col", 1)
    xdot_col = ca.SX.sym("xdot_col", 1)
    var_col = Variables(state_variables={'x_col': x_col, 'xdot_col': xdot_col})
    lag_col = CollisionLagrangian(var_col)
    geo_col = CollisionGeometry(var_col, lam=0.1)
    fks = [x_ee, x_f]
    r_body = 0.1
    for obst in obsts:
        for fk in fks:
            dm_col = CollisionMap(var_q, fk, obst.position(), obst.radius() + r_body)
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
    dm_psi, lag_psi, _, var_psi = defaultAttractor(var_q, goal.position(), fk_ee)
    geo_psi = GoalGeometry(var_psi, k_psi=5)
    planner.addForcingGeometry(dm_psi, lag_psi, geo_psi)
    # finalize
    exLag = ExecutionLagrangian(var_q)
    planner.setExecutionEnergy(exLag)
    exLag.concretize()
    # Speed control
    ex_factor = 1.0
    planner.setDefaultSpeedControl(
        var_psi.position_variable(), dm_psi, exLag, ex_factor,
        #r_b=0.5, b=[0.2, 15.0]
    )
    planner.concretize()
    # running the simulation
    env = gym.make("ground-robot-acc-v0", dt=0.010, render=True)
    ob = env.reset(pos=np.array([-5.0, 0.0, 0.0]), vel=np.array([1.0, 0.0]))
    for obst in obsts:
        env.add_obstacle(obst)
    env.add_goal(goal)
    print("Starting episode")
    for i in range(n_steps):
        if i % 1000 == 0:
            print("time step : ", i)
        x = ob['x']
        xdot = ob['xdot']
        qdot = ob['vel']
        t0 = time.perf_counter()
        action = planner.computeAction(x=x, xdot=xdot, qudot=qdot)
        t1 = time.perf_counter()
        print(f"computation time in ms: {(t1 - t0)*1000}")
        ob, reward, done, info = env.step(action)


if __name__ == "__main__":
    n_steps = 70000
    ground_robot(n_steps=n_steps)
