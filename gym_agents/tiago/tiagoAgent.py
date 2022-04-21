import gym

# import nLinkUrdfReacher
import time
import os
import sys
import numpy as np
import casadi as ca

from fabrics.planner.nonHolonomicPlanner import DefaultNonHolonomicPlanner
from fabrics.defaults.default_geometries import (
    CollisionGeometry,
    LimitGeometry,
    GoalGeometry,
)
from fabrics.defaults.default_energies import CollisionLagrangian, ExecutionLagrangian
from fabrics.defaults.default_maps import CollisionMap, UpperLimitMap, LowerLimitMap
from fabrics.defaults.default_leaves import defaultAttractor

from fabrics.helpers.variables import Variables

import urdfenvs.tiago_reacher
from MotionPlanningEnv.sphereObstacle import SphereObstacle
from MotionPlanningGoal.staticSubGoal import StaticSubGoal

from forwardkinematics.urdfFks.tiagoFk import TiagoFk


def tiagoFabric(n_steps=1000, render=True):
    ## setting up the problem
    n = 20
    obst1Dict = {
        "dim": 3,
        "type": "sphere",
        "geometry": {"position": [0.5, 0.0, 0.0], "radius": 0.2},
    }
    obsts = [
        SphereObstacle(name="obst1", contentDict=obst1Dict),
    ]
    planner = DefaultNonHolonomicPlanner(n, m_base=1, m_arm=0.01)
    var_q = planner.var()
    tiagoFk = TiagoFk()
    # collision avoidance
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    var_x = Variables(state_variables={'x': x, 'xdot': xdot})
    lag_col = CollisionLagrangian(var_x)
    geo_col = CollisionGeometry(var_x, exp=3, lam=1)
    for i in range(1, n + 1):
        fk = tiagoFk.fk(var_q.position_variable(), i, positionOnly=True)
        for obst in obsts:
            dm_col = CollisionMap(var_q, fk, obst.position(), obst.radius())
            # planner.addGeometry(dm_col, lag_col, geo_col)

    # joint limits
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    lag_col = CollisionLagrangian(var_x)
    geo_col = LimitGeometry(var_x, lam=0.5, exp=2)
    limits = [
        [
            -10.0,
            -10.0,
            -6.28318531,
            0.0,
            -1.30899694,
            -1.04719755,
            -1.17809725,
            -1.17809725,
            -0.78539816,
            -0.39269908,
            -2.0943951,
            -1.41371669,
            -2.0943951,
            -1.17809725,
            -1.17809725,
            -0.78539816,
            -0.39269908,
            -2.0943951,
            -1.41371669,
            -2.0943951,
        ],
        [
            10.0,
            10.0,
            6.28318531,
            0.35,
            1.30899694,
            0.78539816,
            1.57079633,
            1.57079633,
            3.92699082,
            2.35619449,
            2.0943951,
            1.41371669,
            2.0943951,
            1.57079633,
            1.57079633,
            3.92699082,
            2.35619449,
            2.0943951,
            1.41371669,
            2.0943951,
        ],
    ]
    for i in range(3, n):
        dm_col = UpperLimitMap(var_q,limits[1][i], i)
        planner.addGeometry(dm_col, lag_col, geo_col)
        dm_col = LowerLimitMap(var_q, limits[0][i], i)
        planner.addGeometry(dm_col, lag_col, geo_col)
    goal1Dict = {
        "m": 3,
        "w": 1.0,
        "prime": True,
        "indices": [0, 1, 2],
        "parent_link": 0,
        "child_link": 11,
        "desired_position": [-0.1, 0.8, 0.8],
        "epsilon": 0.05,
        "type": "staticSubGoal",
    }
    goal1 = StaticSubGoal(name="goal1", contentDict=goal1Dict)
    fk_ee1 = tiagoFk.fk(var_q.position_variable(), goal1.childLink(), positionOnly=True)
    dm_psi1, lag_psi1, _, var_psi1 = defaultAttractor(
        var_q, goal1.position(), fk_ee1
    )
    geo_psi1 = GoalGeometry(var_psi1, k_psi=10)
    goal2Dict = {
        "m": 3,
        "w": 1.0,
        "prime": True,
        "indices": [0, 1, 2],
        "parent_link": 0,
        "child_link": 18,
        "desired_position": [0.2, -0.5, 0.5],
        "epsilon": 0.05,
        "type": "staticSubGoal",
    }
    goal2 = StaticSubGoal(name="goal2", contentDict=goal2Dict)
    fk_ee2 = tiagoFk.fk(var_q.position_variable(), goal2.childLink(), positionOnly=True)
    dm_psi2, lag_psi2, _, var_psi2 = defaultAttractor(
        var_q, goal2.position(), fk_ee2
    )
    geo_psi2 = GoalGeometry(var_psi2, k_psi=5)
    # adding forcing terms
    planner.addForcingGeometry(dm_psi1, lag_psi1, geo_psi1)
    planner.addForcingGeometry(dm_psi2, lag_psi2, geo_psi2)
    # execution energy
    exLag = ExecutionLagrangian(var_q)
    planner.setExecutionEnergy(exLag)
    # speed control
    ex_factor = 1.0
    planner.setDefaultSpeedControl(var_psi1.position_variable(), dm_psi1, exLag, ex_factor, r_b=0.2)
    planner.concretize()
    ## running the simulation
    env = gym.make("tiago-reacher-acc-v0", dt=0.01, render=render)
    print("Starting episode")
    q0 = np.zeros(20)
    q0[3] = 0.1
    q0[6] = 1
    q0[7] = 0
    q0[8] = 2
    q0[9] = 1
    q0[13] = 1
    q0[14] = 1
    q0[16] = 1
    ob = env.reset(pos=q0)
    env.add_goal(goal1)
    env.add_goal(goal2)
    for i in range(n_steps):
        qudot = np.concatenate((ob["vel"], ob["xdot"][3:]))
        action = planner.computeAction(x=ob["x"], xdot=ob["xdot"], qudot=qudot)
        ob, _, _, _ = env.step(action)
    env.close()
    return {}


if __name__ == "__main__":
    res = tiagoFabric(n_steps=10000, render=True)
