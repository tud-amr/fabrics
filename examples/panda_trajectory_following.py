import gym
import sys
import os
import urdfenvs.panda_reacher  #pylint: disable=unused-import

from MotionPlanningGoal.goalComposition import GoalComposition
from MotionPlanningEnv.sphereObstacle import SphereObstacle

import numpy as np
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner


def initalize_environment(render=True):
    """
    Initializes the simulation environment.

    Adds obstacles and goal visualizaion to the environment based and
    steps the simulation once.
    """
    env = gym.make("panda-reacher-acc-v0", dt=0.05, render=render)
    initial_observation = env.reset()
    # Definition of the goal.
    spline_goal = True
    if spline_goal:
        goal_dict = {
            "subgoal0": {
                "weight": 0.4,
                "is_primary_goal": True,
                "indices": [0, 1, 2],
                "parent_link": "panda_link0",
                "child_link": "panda_hand",
                "trajectory": {
                    "controlPoints": [[0.5, 0.5, 0.9], [0.5, 0.0, 0.7], [0.5, -0.5, 0.6]], 
                    "degree": 2,
                    "duration": 10,
                    "low": {
                        "controlPoints": [[0.2, 0.2, 0.5], [0.2, 0.1, 0.5], [0.2, -0.7, 0.5]],
                    },
                    "high": {
                        "controlPoints": [[0.7, 0.7, 0.9], [0.7, -0.1, 0.9], [0.7, -0.2, 0.9]],
                    },
                },
                "epsilon": 0.02,
                "type": "splineSubGoal",
            },
        }
    else:
        goal_dict = {
            "subgoal0": {
                "weight": 0.4,
                "is_primary_goal": True,
                "indices": [0, 1, 2],
                "parent_link": "panda_link0",
                "child_link": "panda_hand",
                "trajectory": ["0.5 + 0.1 * ca.cos(0.2 * t)", "-0.6 * ca.sin(0.2 * t)", "0.4"],
                "epsilon": 0.02,
                "type": "analyticSubGoal",
            },
        }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    goal.shuffle()
    env.add_goal(goal)
    return (env, goal, initial_observation)


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

    ## Optional reconfiguration of the planner
    # base_inertia = 0.03
    # attractor_potential = "20 * ca.norm_2(x)**4"
    # damper = {
    #     "alpha_b": 0.5,
    #     "alpha_eta": 0.5,
    #     "alpha_shift": 0.5,
    #     "beta_distant": 0.01,
    #     "beta_close": 6.5,
    #     "radius_shift": 0.1,
    # }
    # planner = ParameterizedFabricPlanner(
    #     degrees_of_freedom,
    #     robot_type,
    #     base_inertia=base_inertia,
    #     attractor_potential=attractor_potential,
    #     damper=damper,
    # )
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    with open(absolute_path + "/panda.urdf", "r") as file:
        urdf = file.read()
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        'panda',
        urdf=urdf,
        root_link='panda_link0',
        end_link='panda_link9',
    )
    # The planner hides all the logic behind the function set_components.
    collision_links = ['panda_link9', 'panda_link8', 'panda_link4']
    panda_limits = [
            [-2.8973, 2.8973],
            [-1.7628, 1.7628],
            [-2.8973, 2.8973],
            [-3.0718, -0.0698],
            [-2.8973, 2.8973],
            [-0.0175, 3.7525],
            [-2.8973, 2.8973]
        ]
    self_collision_pairs = {}
    planner.set_components(
        collision_links,
        self_collision_pairs,
        goal,
        number_obstacles=0,
        limits=panda_limits,
    )
    planner.concretize()
    return planner


def run_panda_trajectory_example(n_steps=5000, render=True, dynamic_fabric: bool = True):
    print(f"Running example with dynamic fabrics? {dynamic_fabric}")
    (env, goal, initial_observation) = initalize_environment(
        render=render
    )
    ob = initial_observation
    planner = set_planner(goal)

    # Start the simulation
    print("Starting simulation")
    sub_goal_0_weight= np.array(goal.sub_goals()[0].weight())
    for _ in range(n_steps):
        sub_goal_0_position = np.array(goal.sub_goals()[0].position(t=env.t()))
        sub_goal_0_velocity = np.array(goal.sub_goals()[0].velocity(t=env.t()))
        sub_goal_0_acceleration = np.array(goal.sub_goals()[0].acceleration(t=env.t()))
        if not dynamic_fabric:
            sub_goal_0_velocity *= 0
            sub_goal_0_acceleration *= 0
        action = planner.compute_action(
            q=ob["joint_state"]["position"],
            qdot=ob["joint_state"]["velocity"],
            x_ref_goal_0_leaf=sub_goal_0_position,
            xdot_ref_goal_0_leaf=sub_goal_0_velocity,
            xddot_ref_goal_0_leaf=sub_goal_0_acceleration,
            weight_goal_0=sub_goal_0_weight,
        )
        ob, *_ = env.step(action)
    return {}


if __name__ == "__main__":
    arguments = sys.argv
    if len(arguments) == 1 or arguments[0] == 'dynamic_fabric':
        dynamic_fabric = True
    else:
        dynamic_fabric = False
    res = run_panda_trajectory_example(n_steps=5000, dynamic_fabric=dynamic_fabric)
