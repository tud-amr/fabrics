import gym
import sys
import os
import urdfenvs.panda_reacher  #pylint: disable=unused-import

from MotionPlanningGoal.goalComposition import GoalComposition
from MotionPlanningEnv.sphereObstacle import SphereObstacle
from forwardkinematics.urdfFks.pandaFk import PandaFk

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
    goal_dict = {
        "subgoal0": {
            "m": 3,
            "w": 0.4,
            "prime": True,
            "indices": [0, 1, 2],
            "parent_link": "panda_link0",
            "child_link": "panda_hand",
            "trajectory": ["0.5 + 0.1 * ca.cos(0.2 * t)", "-0.6 * ca.sin(0.2 * t)", "0.4"],
            "epsilon": 0.02,
            "type": "analyticSubGoal",
        },
    }
    goal = GoalComposition(name="goal", contentDict=goal_dict)
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
    q = planner.variables.position_variable()
    panda_fk = PandaFk()
    forward_kinematics = []
    for i in range(1, degrees_of_freedom+1):
        forward_kinematics.append(panda_fk.fk(q, i, positionOnly=True))
    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        forward_kinematics,
        goal,
        number_obstacles=0,
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
    sub_goal_0_weight= np.array(goal.subGoals()[0].weight())
    for _ in range(n_steps):
        sub_goal_0_position = np.array(goal.subGoals()[0].position(t=env.t()))
        sub_goal_0_velocity = np.array(goal.subGoals()[0].velocity(t=env.t()))
        sub_goal_0_acceleration = np.array(goal.subGoals()[0].acceleration(t=env.t()))
        if not dynamic_fabric:
            sub_goal_0_velocity *= 0
            sub_goal_0_acceleration *= 0
        action = planner.compute_action(
            q=ob["x"],
            qdot=ob["xdot"],
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
