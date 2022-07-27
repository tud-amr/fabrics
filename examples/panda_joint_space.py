import gym
import urdfenvs.panda_reacher  #pylint: disable=unused-import
import os

from MotionPlanningGoal.goalComposition import GoalComposition

import numpy as np
import os
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
    goal_dict = {
        "subgoal0": {
            "m": 7,
            "w": 0.5,
            "prime": True,
            "indices": list(range(0,7)),
            "desired_position": [-1.0, 0.7, 0.5, -1.501, 0.0, 1.8675, 0.0],
            "epsilon": 0.05,
            "type": "staticJointSpaceSubGoal",
        }
    }
    goal = GoalComposition(name="goal", contentDict=goal_dict)
    env.add_goal(goal)
    return (env, (), goal, initial_observation)


def set_planner(goal: GoalComposition, degrees_of_freedom: int = 7):
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
    collision_links = ['panda_link9', 'panda_link3', 'panda_link4']
    self_collision_pairs = {}
    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        collision_links,
        self_collision_pairs,
        goal,
        number_obstacles=0,
    )
    planner.concretize()
    return planner


def run_panda_joint_space(n_steps=5000, render=True):
    (env, _, goal, initial_observation) = initalize_environment(
        render=render
    )
    ob = initial_observation
    planner = set_planner(goal)

    # Start the simulation
    print("Starting simulation")
    sub_goal_0_position = np.array(goal.subGoals()[0].position())
    sub_goal_0_weight= np.array(goal.subGoals()[0].weight())
    for _ in range(n_steps):
        action = planner.compute_action(
            q=ob["joint_state"]["position"],
            qdot=ob["joint_state"]["velocity"],
            x_goal_0=sub_goal_0_position,
            weight_goal_0=sub_goal_0_weight,
        )
        ob, *_ = env.step(action)
    return {}


if __name__ == "__main__":
    res = run_panda_joint_space(n_steps=5000)
