import gym
import planarenvs.n_link_reacher
import time
import casadi as ca
import numpy as np

from fabrics.planner.fabricPlanner import DefaultFabricPlanner
from fabrics.defaults.default_geometries import CollisionGeometry, LimitGeometry, GoalGeometry
from fabrics.defaults.default_energies import CollisionLagrangian, ExecutionLagrangian
from fabrics.defaults.default_maps import CollisionMap, UpperLimitMap, LowerLimitMap
from fabrics.defaults.default_leaves import defaultAttractor

from fabrics.helpers.variables import Variables

from MotionPlanningEnv.sphereObstacle import SphereObstacle
from MotionPlanningGoal.staticSubGoal import StaticSubGoal

from forwardkinematics.planarFks.planarArmFk import PlanarArmFk


def nlink(n=3, n_steps=5000, render=True):
    # setting up the problem
    obst1Dict = {'dim': 2, 'type': 'sphere', 'geometry': {'position': [2.0, 4.0], 'radius': 0.5}} 
    obst2Dict = {'dim': 2, 'type': 'sphere', 'geometry': {'position': [0.0, 3.0], 'radius': 0.5}} 
    obsts = [
        SphereObstacle(name="obst1", contentDict=obst1Dict),
        SphereObstacle(name="obst2", contentDict=obst2Dict),
    ]
    planner = DefaultFabricPlanner(n)
    var_q = planner.var()
    planarArmFk = PlanarArmFk(n)
    # collision avoidance
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    var_x = Variables(state_variables={'x': x, 'xdot': xdot})
    lag_col = CollisionLagrangian(var_x)
    geo_col = CollisionGeometry(var_x, exp=3, lam=1)
    q = var_q.position_variable()
    for i in range(1, n+1):
        fk = planarArmFk.fk(q, i, positionOnly=True)
        for obst in obsts:
            dm_col = CollisionMap(var_q, fk, obst.position(), obst.radius())
            planner.addGeometry(dm_col, lag_col, geo_col)
    # joint limit avoidance
    lag_lim = CollisionLagrangian(var_x)
    geo_lim = LimitGeometry(var_x, lam=4.0, exp=2)
    for i in range(n):
        dm_lim_upper = UpperLimitMap(var_q, 1.0 * np.pi, i)
        planner.addGeometry(dm_lim_upper, lag_lim, geo_lim)
        dm_lim_lower = LowerLimitMap(var_q, -1.0 * np.pi, i)
        planner.addGeometry(dm_lim_lower, lag_lim, geo_lim)
    # forcing term
    goalDict = {
        "m": 2,
        "w": 1.0,
        "prime": True,
        "indices": [0, 1],
        "parent_link": 0,
        "child_link": 3,
        "desired_position": [-2.0, 2.0],
        "epsilon": 0.2,
        "type": "staticSubGoal",
    }
    goal = StaticSubGoal(name='goal', contentDict=goalDict)
    fk_ee = planarArmFk.fk(q, n, positionOnly=True)
    dm_psi, lag_psi, geo_psi, var_psi = defaultAttractor(var_q, goal.position(), fk_ee)
    geo_psi = GoalGeometry(var_psi, k_psi=2)
    planner.addForcingGeometry(dm_psi, lag_psi, geo_psi)
    # execution energy
    exLag = ExecutionLagrangian(var_q)
    planner.setExecutionEnergy(exLag)
    # Speed control
    ex_factor = 1.0
    planner.setDefaultSpeedControl(var_psi.position_variable(), dm_psi, exLag, ex_factor, r_b=0.2)
    planner.concretize()
    # setup environment
    # running the simulation
    env = gym.make("nLink-reacher-acc-v0", n=n, dt=0.05, render=render)
    print("Starting episode")
    q0 = np.zeros(n)
    q0dot = np.array([0.1, 0.4, 0.1, 0.0, 0.0])
    ob = env.reset(pos=q0, vel=q0dot)
    for obst in obsts:
        env.add_obstacle(obst)
    env.add_goal(goal)
    for i in range(n_steps):
        action = planner.computeAction(q=ob['x'], qdot=ob['xdot'])
        # env.render()
        ob, reward, done, info = env.step(action)
    return {}

if __name__ == "__main__":
    n_steps = 5000
    n = 5
    nlink(n=n, n_steps=n_steps)
