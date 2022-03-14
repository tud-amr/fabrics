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

import urdfenvs.pandaReacher
from MotionPlanningEnv.sphereObstacle import SphereObstacle
from MotionPlanningGoal.staticSubGoal import StaticSubGoal

from forwardkinematics.urdfFks.pandaFk import PandaFk


def pandaFabric(n_steps=1000, render=True):
    ## setting up the problem
    n = 7
    obst1Dict = {'dim': 3, 'type': 'sphere', 'geometry': {'position': [0.5, 0.0, 0.0], 'radius': 0.2}} 
    obsts = [
        SphereObstacle(name="obst1", contentDict=obst1Dict),
    ]
    planner = DefaultFabricPlanner(n)
    q, qdot = planner.var()
    pandaFk = PandaFk(n)
    # collision avoidance
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    lag_col = CollisionLagrangian(x, xdot)
    geo_col = CollisionGeometry(x, xdot, exp=3, lam=1)
    for i in range(1, n+1):
        fk = pandaFk.fk(q, i, positionOnly=True)
        for obst in obsts:
            dm_col = CollisionMap(q, qdot, fk, obst.position(), obst.radius())
            planner.addGeometry(dm_col, lag_col, geo_col)
    goalDict = {
        "m": 3,
        "w": 1.0,
        "prime": True,
        "indices": [0, 1, 2],
        "parent_link": 0,
        "child_link": 3,
        "desired_position": [0.4, -0.4, 0.3],
        "epsilon": 0.02,
        "type": "staticSubGoal",
    }
    goal = StaticSubGoal(name='goal', contentDict=goalDict)
    fk_ee = pandaFk.fk(q, n, positionOnly=True)
    dm_psi, lag_psi, geo_psi, x_psi, xdot_psi = defaultAttractor(q, qdot, goal.position(), fk_ee)
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
    env = gym.make('panda-reacher-acc-v0', dt=0.01, render=render, gripper=False)
    print("Starting episode")
    q0 = np.array([0.8, 0.7, 0.0, -1.501, 0.0, 1.8675, 0.0])
    env.addObstacle(obsts[0])
    env.addGoal(goal)
    ob = env.reset(pos=q0)
    for i in range(n_steps):
        action = planner.computeAction(ob['x'], ob['xdot'])
        ob, reward, done, info = env.step(action)
    return {}


if __name__ == "__main__":
    res = pandaFabric(n_steps=10000, render=True)
