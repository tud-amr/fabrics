import gym
import urdfenvs.panda_reacher  #pylint: disable=unused-import

from MotionPlanningGoal.goalComposition import GoalComposition
from MotionPlanningEnv.sphereObstacle import SphereObstacle
from forwardkinematics.urdfFks.pandaFk import PandaFk

import numpy as np
from fabrics.planner.serialized_planner import SerializedFabricPlanner

import os
import casadi as ca
import pickle


# serialized file name
# 10 denotes 10 spheres
serialized_file = "serialized_10.pkl"

# boolean to determine if the serialized file has been loaded
isload = False

def initalize_environment(render=True, obstacle_resolution = 8):
    """
    Initializes the simulation environment.

    Adds obstacles and goal visualizaion to the environment based and
    steps the simulation once.
    """
    env = gym.make("panda-reacher-acc-v0", dt=0.05, render=render)
    q0 = np.array([0.0, -1.0, 0.0, -1.501, 0.0, 1.8675, 0.0])
    initial_observation = env.reset(pos=q0)
    # Definition of the obstacle.
    radius_ring = 0.20
    obstacles = []
    whole_position = [0.4, 0.0, 0.7]
    for i in range(obstacle_resolution + 1):
        angle = i/obstacle_resolution * 2.*np.pi
        position = [
            whole_position[0],
            whole_position[1] + radius_ring * np.cos(angle),
            whole_position[2] + radius_ring * np.sin(angle),
        ]
        static_obst_dict = {
            "dim": 3,
            "type": "sphere",
            "geometry": {"position": position, "radius": 0.05},
        }
        obstacles.append(SphereObstacle(name="staticObst", contentDict=static_obst_dict))
    # Definition of the goal.
    goal_position = whole_position
    goal_position[0] += 0.3
    goal_dict = {
        "subgoal0": {
            "m": 3,
            "w": 200.0,
            "prime": True,
            "indices": [0, 1, 2],
            "parent_link": 0,
            "child_link": 7,
            "desired_position": whole_position,
            "epsilon": 0.05,
            "type": "staticSubGoal",
        },
        "subgoal1": {
            "m": 2,
            "w": 5000.0,
            "prime": False,
            "indices": [1, 2],
            "parent_link": 6,
            "child_link": 7,
            "desired_position": [0.0, 0.0],
            "epsilon": 0.05,
            "type": "staticSubGoal",
        }
    }
    goal = GoalComposition(name="goal", contentDict=goal_dict)
    env.add_goal(goal)
    for obst in obstacles:
        env.add_obstacle(obst)
    return (env, obstacles, goal, initial_observation)


def set_planner(goal: GoalComposition, degrees_of_freedom: int = 7, obstacle_resolution = 10):
    """
    Initializes the fabric planner for the panda robot.

    This function defines the forward kinematics for collision avoidance,
    and goal reaching. These components are fed into the fabrics planner.

    In the top section of this function, an example for optional reconfiguration
    can be found. Commented by default.

    Params
    ----------
    goal: StaticSubGoal
        The goal to the motion planning problem.
    degrees_of_freedom: int
        Degrees of freedom of the robot (default = 7)
    """

    robot_type = 'panda'


    attractor_potential = "10 * ca.norm_2(x)**2"
    collision_finsler = "2.0/(x ** 2) * xdot ** 2"
    planner = SerializedFabricPlanner(
        serialized_file,
    )

    if not planner._isload:
        print(f"Serializing planner function and input_keys to {serialized_file}")
        q = planner.variables.position_variable()
        panda_fk = PandaFk()
        forward_kinematics = []
        for i in range(1, degrees_of_freedom+1):
            forward_kinematics.append(panda_fk.fk(q, i, positionOnly=True))
        # The planner hides all the logic behind the function set_components.
        planner.set_components(
            forward_kinematics,
            goal,
            number_obstacles=obstacle_resolution,
        )
        # concertize here
        planner.serialize(serialized_file)

    return planner


def run_panda_ring_example(n_steps=5000, render=True):
    obstacle_resolution_ring = 10
    (env, obstacles, goal, initial_observation) = initalize_environment(
        render=render, obstacle_resolution=obstacle_resolution_ring
    )
    ob = initial_observation
    planner = set_planner(goal, obstacle_resolution= obstacle_resolution_ring)

    # Start the simulation
    print("Starting simulation")
    sub_goal_0_position = np.array(goal.subGoals()[0].position())
    sub_goal_0_weight= np.array(goal.subGoals()[0].weight())
    sub_goal_1_position = np.array(goal.subGoals()[1].position())
    sub_goal_1_weight= np.array(goal.subGoals()[1].weight())
    obstacle_positions = []
    obstacle_radii = []
    for obst in obstacles:
        obstacle_positions.append(obst.position())
        obstacle_radii.append(np.array(obst.radius()))

    # if serialized_file exists and is loaded
    # pass loaded data [planner._funs, planner._input_keys] to serialized_compute_action
    # otherwise use the compute_action() from ParameterizedFabricPlanner
    if os.path.isfile(serialized_file) and planner._isload:
        for _ in range(n_steps):
            action = planner.compute_action(
                q=ob["x"],
                qdot=ob["xdot"],
                x_goal_0=sub_goal_0_position,
                weight_goal_0=sub_goal_0_weight,
                x_goal_1=sub_goal_1_position,
                weight_goal_1=sub_goal_1_weight,
                x_obsts = obstacle_positions,
                radius_obsts = obstacle_radii,
                radius_body=np.array([0.02]),
            )
            ob, *_ = env.step(action)
    else:
        for _ in range(n_steps):
            action = planner.compute_action(
                q=ob["x"],
                qdot=ob["xdot"],
                x_goal_0=sub_goal_0_position,
                weight_goal_0=sub_goal_0_weight,
                x_goal_1=sub_goal_1_position,
                weight_goal_1=sub_goal_1_weight,
                x_obsts = obstacle_positions,
                radius_obsts = obstacle_radii,
                radius_body=np.array([0.02]),
            )
            ob, *_ = env.step(action)
    return {}


if __name__ == "__main__":
    res = run_panda_ring_example(n_steps=5000)
