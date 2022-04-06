import gym
import planarenvs.point_robot
from planarenvs.point_robot.envs.acc import PointRobotAccEnv
import time
import casadi as ca
import numpy as np

from fabrics.planner.fabricPlanner import DefaultFabricPlanner
from fabrics.planner.default_geometries import CollisionGeometry, GoalGeometry
from fabrics.planner.default_energies import (
    CollisionLagrangian,
    ExecutionLagrangian,
    GoalLagrangian,
)
from fabrics.planner.default_maps import CollisionMap
from fabrics.planner.default_leaves import defaultAttractor
from fabrics.leaves.attractor import Attractor, ParameterizedAttractor
from fabrics.leaves.obstacle_leaf import ObstacleLeaf

from fabrics.helpers.variables import Variables

from MotionPlanningEnv.sphereObstacle import SphereObstacle
from MotionPlanningGoal.staticSubGoal import StaticSubGoal
from MotionPlanningGoal.dynamicSubGoal import DynamicSubGoal

from planarenvs.sensors.GoalSensor import GoalSensor

# TODO: Currently, the goal sensor is restricted to observations up to the
# limits. Therefore, the robot cannot reach the actual goal which exceeds this
# limit <04-04-22, mspahn> #

def pointMass(n_steps=5000, render=True):
    planner = DefaultFabricPlanner(2)
    var_q = planner.var()
    # collision avoidance
    staticObstDict = {
        "dim": 2,
        "type": "sphere",
        "geometry": {"position": [0.0, 0.0], "radius": 1.0},
    }
    obsts = [
        SphereObstacle(name="staticObst", contentDict=staticObstDict),
    ]
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    var_x = Variables(state_variables={'x': x, 'xdot': xdot})
    lag_col = CollisionLagrangian(var_x)
    geo_col = CollisionGeometry(var_x, exp=2.0)
    fks = [var_q.position_variable()]
    for fk in fks:
        for obst in obsts:
            dm_col = CollisionMap(var_q, fk, obst.position(), obst.radius())
            obstacleLeaf = ObstacleLeaf(var_q)
            obstacleLeaf.set_map(var_q.position_variable(), obst.position(), obst.radius())
            obstacleLeaf.set_params(exp=1.0, lam=1.00)
            obstacleLeaf.concretize()
            planner.add_leaf(obstacleLeaf)
    # forcing term
    goalDict = {
        "m": 2,
        "w": 1.0,
        "prime": True,
        "indices": [0, 1],
        "parent_link": 0,
        "child_link": 1,
        "trajectory": ["-4.0+ 1.1 * t", "0.1"],
        "epsilon": 0.2,
        "type": "analyticSubGoal",
    }
    goal = DynamicSubGoal(name="goal", contentDict=goalDict)
    # TODO: Potentially, you have to either pass the reference variable or return it. <04-04-22, mspahn> #
    attractor = ParameterizedAttractor(var_q)
    #attractor.set_goal(goal.position(), q)
    attractor.set_goal(ca.SX.sym("x_goal", 2), fks[0])
    attractor.set_params(k_psi=5.0)
    attractor.concretize()
    planner.add_leaf(attractor)
    # execution energy
    exLag = ExecutionLagrangian(var_q)
    exLag.concretize()
    planner.setExecutionEnergy(exLag)
    # Speed control
    ex_factor = 1.0
    planner.setDefaultSpeedControl(attractor.x_psi(), attractor.map(), exLag, ex_factor)
    planner.concretize()
    # running the simulation
    x0 = np.array([4.3, -1.0])
    xdot0 = np.array([-1.0, 0.0])
    env: PointRobotAccEnv = gym.make("point-robot-acc-v0", dt=0.01, render=render)
    ob = env.reset(pos=x0, vel=xdot0)
    sensor = GoalSensor(nbGoals=1)
    env.addSensor(sensor)
    env.add_goal(goal)
    env.add_obstacle(obsts[0])
    env.resetLimits(pos={'low':np.array([-1, -2]), 'high':np.array([20, 2])})
    print("Starting episode")
    q = np.zeros((n_steps, 2))
    t = 0.0
    solverTime = np.zeros(n_steps)
    goalPosition = np.array(goal.position())
    for i in range(n_steps):
        action = planner.computeAction(
                q=ob["x"],
                qdot=ob["xdot"],
                x_goal=goalPosition
            )
        ob, reward, done, info = env.step(action)
        ob, _, _, _ = env.step(action)
        goalPosition = ob["GoalPosition"][0]
        print(f"goalPosition : {goalPosition}")
    return {}


if __name__ == "__main__":
    n_steps = 5000
    res = pointMass(n_steps=n_steps)
