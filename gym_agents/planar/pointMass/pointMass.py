import gym
import planarenvs.point_robot
import time
import casadi as ca
import numpy as np

from fabrics.planner.fabricPlanner import DefaultFabricPlanner
from fabrics.planner.default_geometries import CollisionGeometry, GoalGeometry
from fabrics.planner.default_energies import CollisionLagrangian, ExecutionLagrangian, GoalLagrangian
from fabrics.planner.default_maps import CollisionMap
from fabrics.planner.default_leaves import defaultAttractor

from fabrics.helpers.variables import Variables

from MotionPlanningEnv.sphereObstacle import SphereObstacle


def pointMass(n_steps=5000, render=True):
    ## setting up the problem
    staticObstDict = {'dim': 2, 'type': 'sphere', 'geometry': {'position': [0.0, 0.0], 'radius': 1.0}} 
    obsts = [
        SphereObstacle(name="staticObst", contentDict=staticObstDict),
    ]
    n = 2
    planner = DefaultFabricPlanner(n)
    var_q = planner.var()
    # collision avoidance
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    var_x = Variables(state_variables={'x': x, 'xdot': xdot})
    lag_col = CollisionLagrangian(var_x)
    geo_col = CollisionGeometry(var_x, exp=2.0)
    fks = [var_q.position_variable()]
    for fk in fks:
        for obst in obsts:
            dm_col = CollisionMap(var_q, fk, obst.position(), obst.radius())
            planner.addGeometry(dm_col, lag_col, geo_col)
    # forcing term
    q_d = np.array([-2.0, -0.1])
    dm_psi, lag_psi, _, var_psi  = defaultAttractor(var_q, q_d, fk)
    geo_psi = GoalGeometry(var_psi, k_psi=5)
    planner.addForcingGeometry(dm_psi, lag_psi, geo_psi)
    # execution energy
    exLag = ExecutionLagrangian(var_q)
    exLag.concretize()
    planner.setExecutionEnergy(exLag)
    # Speed control
    ex_factor = 1.0
    planner.setDefaultSpeedControl(var_psi.position_variable(), dm_psi, exLag, ex_factor)
    planner.concretize()
    # setup environment
    x0 = np.array([4.3, 0.8])
    xdot0 = np.array([-1.0, -0.0])
    # running the simulation
    env = gym.make('point-robot-acc-v0', dt=0.01, render=render)
    ob = env.reset(pos=x0, vel=xdot0)
    for obst in obsts:
        env.add_obstacle(obst)
    print("Starting episode")
    for _ in range(n_steps):
        action = planner.computeAction(q=ob['x'], qdot=ob['xdot'])
        ob, _, _, _ = env.step(action)
    ## Plotting the results
    return {}

if __name__ == "__main__":
    n_steps = 5000
    res = pointMass(n_steps=n_steps)
