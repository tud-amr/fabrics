import gym
# import nLinkUrdfReacher
import time
import os
import sys
import numpy as np
import casadi as ca

from fabrics.planner.fabricPlanner import DefaultFabricPlanner
from fabrics.planner.default_geometries import CollisionGeometry, LimitGeometry, GoalGeometry
from fabrics.planner.default_energies import CollisionLagrangian, ExecutionLagrangian
from fabrics.planner.default_maps import CollisionMap, UpperLimitMap, LowerLimitMap
from fabrics.planner.default_leaves import defaultAttractor

import urdfenvs.dual_arm
from MotionPlanningEnv.sphereObstacle import SphereObstacle
from MotionPlanningGoal.staticSubGoal import StaticSubGoal

from forwardkinematics.urdfFks.dual_arm_fk import DualArmFk


def dualArmFabric(n_steps=1000, render=True):
    ## setting up the problem
    n = 5
    obst1Dict = {'dim': 3, 'type': 'sphere', 'geometry': {'position': [0.5, 0.0, 0.0], 'radius': 0.2}} 
    obsts = [
        SphereObstacle(name="obst1", contentDict=obst1Dict),
    ]
    planner = DefaultFabricPlanner(n)
    q, qdot = planner.var()
    fk = DualArmFk()
    # collision avoidance
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    lag_col = CollisionLagrangian(x, xdot)
    geo_col = CollisionGeometry(x, xdot, exp=3, lam=1)
    for i in range(1, n+1):
        fk_i = fk.fk(q, i, positionOnly=True)
        for obst in obsts:
            dm_col = CollisionMap(q, qdot, fk_i, obst.position(), obst.radius())
            #planner.addGeometry(dm_col, lag_col, geo_col)
    goalDict1 = {
        "m": 3,
        "w": 1.0,
        "prime": True,
        "indices": [0, 1, 2],
        "parent_link": 0,
        "child_link": 3,
        "desired_position": [-0.5, 2.4, 1.3],
        "epsilon": 0.10,
        "type": "staticSubGoal",
    }
    goal1 = StaticSubGoal(name='goal', contentDict=goalDict1)
    fk_ee = fk.fk(q, goal1.childLink(), positionOnly=True)
    dm_psi, lag_psi, geo_psi, x_psi, xdot_psi = defaultAttractor(q, qdot, goal1.position(), fk_ee)
    geo_psi = GoalGeometry(x_psi, xdot_psi, k_psi=10)
    planner.addForcingGeometry(dm_psi, lag_psi, geo_psi)
    goalDict2 = {
        "m": 3,
        "w": 1.0,
        "prime": True,
        "indices": [0, 1, 2],
        "parent_link": 0,
        "child_link": 5,
        "desired_position": [1.5, -0.4, 2.3],
        "epsilon": 0.10,
        "type": "staticSubGoal",
    }
    goal2 = StaticSubGoal(name='goal', contentDict=goalDict2)
    fk_ee = fk.fk(q, goal2.childLink(), positionOnly=True)
    dm_psi, lag_psi, geo_psi, x_psi, xdot_psi = defaultAttractor(q, qdot, goal2.position(), fk_ee)
    geo_psi = GoalGeometry(x_psi, xdot_psi, k_psi=10)
    planner.addForcingGeometry(dm_psi, lag_psi, geo_psi)
    # execution energy
    exLag = ExecutionLagrangian(q, qdot)
    planner.setExecutionEnergy(exLag)
    # speed control 
    ex_factor = 1.0
    planner.setDefaultSpeedControl(x_psi, dm_psi, exLag, ex_factor, r_b=0.2)
    planner.concretize()
    ## running the simulation
    env = gym.make('dual-arm-acc-v0', dt=0.01, render=render)
    print("Starting episode")
    env.add_obstacle(obsts[0])
    env.add_goal(goal1)
    env.add_goal(goal2)
    ob = env.reset()
    for i in range(n_steps):
        action = planner.computeAction(ob['x'], ob['xdot'])
        ob, reward, done, info = env.step(action)
    return {}


if __name__ == "__main__":
    res = dualArmFabric(n_steps=10000, render=True)
