import gym
import urdfenvs.tiago_reacher #pylint: disable=unused-import
import os
import logging
from copy import deepcopy
from pyquaternion import Quaternion

from MotionPlanningGoal.goalComposition import GoalComposition
from MotionPlanningEnv.sphereObstacle import SphereObstacle

import numpy as np
import os
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

logging.basicConfig(level=logging.DEBUG)

root_link = 'torso_lift_link'
arm = 'left'
child_link = f'gripper_{arm}_grasping_frame'


def initalize_environment(render=True):
    """
    Initializes the simulation environment.

    Adds obstacles and goal visualizaion to the environment based and
    steps the simulation once.
    """
    env = gym.make("tiago-reacher-acc-v0", dt=0.05, render=render)
    pos0 = np.zeros(20)
    pos0[0] = 0.0
    # base
    # pos0[0:3] = np.array([0.0, 1.0, -1.0])
    # torso
    pos0[3] = 0.1
    # Set joint values to center position
    if arm == 'left':
        limits = env.env._robot._limit_pos_j.transpose()[6:13]
    elif arm == 'right':
        limits = env.env._robot._limit_pos_j.transpose()[13:20]
    pos0[13:20] = (limits[:, 0] + limits[:, 1]) / 2.0
    pos0[6:13] = (limits[:, 0] + limits[:, 1]) / 2.0
    initial_observation = env.reset(pos=pos0)
    # Definition of the obstacle.
    static_obst_dict = {
        "dim": 3,
        "type": "sphere",
        "geometry": {"position": [0.8, 0.1, 0.7], "radius": 0.1},
    }
    obst = SphereObstacle(name="staticObst", contentDict=static_obst_dict)
    # Definition of the goal.
    if arm == 'left':
        high_goal_limits = [0.4, 1.0, 1.3]
        low_goal_limits = [-0.4, 0.5, 0.4]
    if arm == 'right':
        high_goal_limits = [0.4, -1.0, 1.3]
        low_goal_limits = [-0.4, -0.5, 0.4]
    goal_dict = {
        "subgoal0": {
            "m": 3,
            "w": 1.0,
            "prime": True,
            "indices": [0, 1, 2],
            "parent_link": root_link,
            "child_link": child_link,
            "desired_position": [0.3, -0.05, 0.8],
            "high": high_goal_limits,
            "low": low_goal_limits,
            "epsilon": 0.05,
            "type": "staticSubGoal",
        },
    }
    # Transform goal and obst into world frame
    goal = GoalComposition(name="goal", contentDict=goal_dict)
    #goal.shuffle()

    goal_transformed_dict = deepcopy(goal_dict)
    desired_position = deepcopy(goal.primeGoal().position())
    desired_position[2] -= 0.99
    desired_position[0] += 0.242
    goal_transformed_dict['subgoal0']['desired_position'] = desired_position
    goal_transformed = GoalComposition(name="goal", contentDict=goal_transformed_dict)
    obst_transformed_dict = deepcopy(static_obst_dict)
    desired_position = deepcopy(obst.position())
    desired_position[2] -= 0.99
    desired_position[0] += 0.242
    obst_transformed_dict['geometry']['position'] = desired_position
    obst_transformed = SphereObstacle(name="obst", contentDict=obst_transformed_dict)
    obstacles = [obst_transformed]
    env.add_goal(goal)
    env.add_obstacle(obst)
    return (env, obstacles, goal_transformed, initial_observation)


def set_planner(goal: GoalComposition, limits: np.ndarray, degrees_of_freedom: int = 7):
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

    robot_type = 'tiago'

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
    with open(absolute_path + "/tiago_dual.urdf", "r") as file:
        urdf = file.read()
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        'tiago',
        urdf=urdf,
        root_link='torso_lift_link',
        end_link=f'gripper_{arm}_link',
    )
    q = planner.variables.position_variable()
    collision_links = [f'arm_{arm}_{i}_link' for i in [3, 4, 5, 6, 7]]
    self_collision_pairs = {}
    # The planner hides all the logic behind the function set_components.
    logging.debug(limits)
    planner.set_components(
        collision_links,
        self_collision_pairs,
        goal,
        limits=list(limits),
        number_obstacles=1,
    )
    planner.concretize()
    return planner


def run_tiago_example(n_steps=5000, render=True):
    (env, obstacles, goal, initial_observation) = initalize_environment(
        render=render
    )
    ob = initial_observation
    obst1 = obstacles[0]
    if arm == 'left':
        limits = env.env._robot._limit_pos_j.transpose()[6:13]
    elif arm == 'right':
        limits = env.env._robot._limit_pos_j.transpose()[13:20]
    planner = set_planner(goal, limits)

    # Start the simulation
    print("Starting simulation")
    sub_goal_0_position = np.array(goal.subGoals()[0].position())
    sub_goal_0_weight= np.array(goal.subGoals()[0].weight())
    obst1_position = np.array(obst1.position())
    augmented_action = np.zeros(19)
    body_arguments = {}
    for i in [3, 4, 5, 6, 7]:
        body_arguments[f'radius_body_arm_{arm}_{i}_link'] =np.array([0.1])
    for _ in range(n_steps):
        if arm == 'left':
            q = ob['x'][6:13]
            qdot = ob['xdot'][6:13]
        elif arm == 'right':
            q = ob['x'][13:20]
            qdot = ob['xdot'][13:20]
        logging.debug(f"q[0]: {q[0]}")
        action = planner.compute_action(
            q=q,
            qdot=qdot,
            x_goal_0=sub_goal_0_position,
            weight_goal_0=sub_goal_0_weight,
            x_obst_0=obst1_position,
            radius_obst_0=np.array([obst1.radius()]),
            **body_arguments
        )
        fk = compute_fk(planner._forward_kinematics.fk, q)
        logging.debug(f"fk : {fk}")
        #logging.debug(f"action : {action}")
        if arm == 'left':
            augmented_action[5:12] = action
        elif arm == 'right':
            augmented_action[12:19] = action
        ob, *_ = env.step(augmented_action)
    return {}

def compute_fk(fk_fun, q):
    return fk_fun(
        q,
        root_link,
        child_link,
        positionOnly=True
    )


if __name__ == "__main__":
    res = run_tiago_example(n_steps=5000)
