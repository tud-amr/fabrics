import gymnasium as gym
import sys
import os
import logging
import numpy as np

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor

from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle

from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner


def initalize_environment(render=True):
    """
    Initializes the simulation environment.

    Adds obstacles and goal visualizaion to the environment based and
    steps the simulation once.
    """
    robots = [
        GenericUrdfReacher(urdf="panda.urdf", mode="acc"),
    ]
    env: UrdfEnv  = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    ).unwrapped
    full_sensor = FullSensor(
            goal_mask=["position", "velocity", "weight"],
            obstacle_mask=[],
            variance=0.0
    )
    # Definition of the goal.
    goal_dict = {
        "subgoal0": {
            "weight": 0.4,
            "is_primary_goal": True,
            "indices": [0, 1, 2],
            "parent_link": "panda_link0",
            "child_link": "panda_hand",
            "trajectory": ["0.5 + 0.1 * sp.cos(0.2 * t)", "-0.6 * sp.sin(0.2 * t)", "0.4"],
            "epsilon": 0.02,
            "type": "analyticSubGoal",
        },
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    env.reset()
    env.add_sensor(full_sensor, [0])
    for sub_goal in goal.sub_goals():
        env.add_goal(sub_goal)
    env.set_spaces()
    return (env, goal)


def set_planner(goal: GoalComposition, degrees_of_freedom: int = 7):
    """
    Initializes the fabric planner for the panda robot.

    This function defines the forward kinematics for collision avoidance,
    and goal reaching. These components are fed into the fabrics planner.

    In the top section of this function, an example for optional reconfiguration
    can be found. Commented by default.

    Params
    ----------
    goal: GoalComposition
        The goal to the motion planning problem. The goal can be composed
        of several subgoals.
    degrees_of_freedom: int
        Degrees of freedom of the robot (default = 7)
    """
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    with open(absolute_path + "/panda_for_fk.urdf", "r", encoding="utf-8") as file:
        urdf = file.read()
    forward_kinematics = GenericURDFFk(
        urdf,
        root_link="panda_link0",
        end_links="panda_link9",
    )
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        forward_kinematics,
    )
    # The planner hides all the logic behind the function set_components.
    collision_links = ['panda_link9', 'panda_link8', 'panda_link4']
    self_collision_pairs = {}
    planner.set_components(
        collision_links=collision_links,
        goal=goal,
        number_obstacles=0,
    )
    planner.concretize()
    return planner


def run_panda_trajectory_example(n_steps=5000, render=True, dynamic_fabric: bool = True):
    (env, goal) = initalize_environment(render)
    planner = set_planner(goal)
    action = np.zeros(7)
    ob, *_ = env.step(action)


    sub_goal_0_acceleration = np.zeros(3)
    logging.warning(f"Running example with dynamic fabrics? {dynamic_fabric}")
    for _ in range(n_steps):
        ob_robot = ob['robot_0']
        sub_goal_0_position = ob_robot['FullSensor']['goals'][2]['position']
        sub_goal_0_velocity = ob_robot['FullSensor']['goals'][2]['velocity']
        if not dynamic_fabric:
            sub_goal_0_velocity *= 0
        action = planner.compute_action(
            q=ob_robot["joint_state"]["position"],
            qdot=ob_robot["joint_state"]["velocity"],
            x_ref_goal_0_leaf=sub_goal_0_position,
            xdot_ref_goal_0_leaf=sub_goal_0_velocity,
            xddot_ref_goal_0_leaf=sub_goal_0_acceleration,
            weight_goal_0=ob_robot['FullSensor']['goals'][2]['weight'],
        )
        ob, *_ = env.step(action)
    env.close()
    return {}


if __name__ == "__main__":
    arguments = sys.argv
    if len(arguments) == 1 or arguments[0] == 'dynamic_fabric':
        dynamic_fabric = True
    else:
        dynamic_fabric = False
    res = run_panda_trajectory_example(n_steps=5000, dynamic_fabric=dynamic_fabric)
