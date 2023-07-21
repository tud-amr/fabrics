import os
import gymnasium as gym
import numpy as np
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher

from mpscenes.goals.goal_composition import GoalComposition

from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

# TODO joint space goals cannot be handled by the full sensor because they cannot
# be added to the environment


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
    )
    # Definition of the goal.
    goal_dict = {
        "subgoal0": {
            "weight": 0.5,
            "is_primary_goal": True,
            "indices": list(range(0,7)),
            "desired_position": [-1.0, 0.7, 0.5, -1.501, 0.0, 1.8675, 0.0],
            "epsilon": 0.05,
            "type": "staticJointSpaceSubGoal",
        }
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    env.reset()
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
    with open(absolute_path + "/panda_for_fk.urdf", "r") as file:
        urdf = file.read()
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        'panda',
        urdf=urdf,
        root_link='panda_link0',
        end_link='panda_link9',
    )
    collision_links = ['panda_link9', 'panda_link3', 'panda_link4']
    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        collision_links=collision_links,
        goal=goal,
        number_obstacles=0,
    )
    planner.concretize()
    return planner


def run_panda_joint_space(n_steps=5000, render=True):
    (env, goal) = initalize_environment(render)
    planner = set_planner(goal)
    action = np.zeros(7)
    ob, *_ = env.step(action)

    for _ in range(n_steps):
        ob_robot = ob['robot_0']
        action = planner.compute_action(
            q=ob_robot["joint_state"]["position"],
            qdot=ob_robot["joint_state"]["velocity"],
            x_goal_0=np.array(goal.sub_goals()[0].position()),
            weight_goal_0=goal.sub_goals()[0].weight(),
        )
        ob, *_ = env.step(action)
    env.close()
    return {}


if __name__ == "__main__":
    res = run_panda_joint_space(n_steps=5000)
