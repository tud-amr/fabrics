# This example is a copy of the example presented in the paper
# "Geometry Fabrics: Generalzing Classical Mechanics to Capture the Physics of Behavior"
# https://arxiv.org/abs/2109.10443
# This implementation is not related to the paper as no code was published with it.
import gym
import os
import urdfenvs.panda_reacher  #pylint: disable=unused-import

from MotionPlanningGoal.goalComposition import GoalComposition
from MotionPlanningEnv.sphereObstacle import SphereObstacle

import numpy as np
import quaternionic
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

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
    radius_ring = 0.3
    obstacles = []
    goal_orientation = [-0.366, 0.0, 0.0, 0.3305]
    rotation_matrix = quaternionic.array(goal_orientation).to_rotation_matrix
    whole_position = [0.1, 0.6, 0.8]
    for i in range(obstacle_resolution + 1):
        angle = i/obstacle_resolution * 2.*np.pi
        origin_position = [
            0.0,
            radius_ring * np.cos(angle),
            radius_ring * np.sin(angle),
        ]
        position = np.dot(np.transpose(rotation_matrix), origin_position) + whole_position
        static_obst_dict = {
            "dim": 3,
            "type": "sphere",
            "geometry": {"position": position.tolist(), "radius": 0.1},
        }
        obstacles.append(SphereObstacle(name="staticObst", contentDict=static_obst_dict))
    # Definition of the goal.
    goal_position = whole_position
    goal_dict = {
        "subgoal0": {
            "m": 3,
            "w": 1.0,
            "prime": True,
            "indices": [0, 1, 2],
            "parent_link": "panda_link0",
            "child_link": "panda_hand",
            "desired_position": whole_position,
            "epsilon": 0.05,
            "type": "staticSubGoal",
        },
        "subgoal1": {
            "m": 3,
            "w": 3.0,
            "prime": False,
            "indices": [0, 1, 2],
            "parent_link": "panda_link7",
            "child_link": "panda_hand",
            "desired_position": [0.107, 0.0, 0.0],
            "angle": goal_orientation,
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

    ## Optional reconfiguration of the planner
    # base_inertia = 0.03
    # attractor_potential = "5.0 * (ca.norm_2(x) + 1 /10 * ca.log(1 + ca.exp(-2 * 10 * ca.norm_2(x))))"
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
    # attractor_potential = "15.0 * (ca.norm_2(x) + 1 /10 * ca.log(1 + ca.exp(-2 * 10 * ca.norm_2(x))))"
    # collision_geometry= "-0.1 / (x ** 2) * (-0.5 * (ca.sign(xdot) - 1)) * xdot ** 2"
    # collision_finsler= "0.1/(x**1) * xdot**2"
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    with open(absolute_path + "/albert_polluted_2.urdf", "r") as file:
        urdf = file.read()
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        robot_type,
        urdf=urdf,
        root_link='panda_link0',
        end_link=['panda_vacuum_2', 'panda_vacuum'],
    )
    panda_limits = [
            [-2.8973, 2.8973],
            [-1.7628, 1.7628],
            [-2.8973, 2.8973],
            [-3.0718, -0.0698],
            [-2.8973, 2.8973],
            [-0.0175, 3.7525],
            [-2.8973, 2.8973]
        ]
    collision_links = ['panda_link1', 'panda_link4', 'panda_link6', 'panda_hand']
    self_collision_pairs = {}
    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        collision_links,
        self_collision_pairs,
        goal,
        number_obstacles=obstacle_resolution,
        limits=panda_limits,
    )
    planner.concretize()
    return planner


def run_panda_ring_example(n_steps=5000, render=True, serialize=False):
    obstacle_resolution_ring = 10
    (env, obstacles, goal, initial_observation) = initalize_environment(
        render=render, obstacle_resolution=obstacle_resolution_ring
    )
    ob = initial_observation
    planner = set_planner(goal, obstacle_resolution= obstacle_resolution_ring)

    # Serializing the planner is optional
    if serialize:
        planner.serialize('serialized_10.pbz2')

    # Start the simulation
    print("Starting simulation")
    sub_goal_0_position = np.array(goal.subGoals()[0].position())
    sub_goal_0_weight = goal.subGoals()[0].weight()
    sub_goal_1_position = np.array(goal.subGoals()[1].position())
    sub_goal_1_weight= goal.subGoals()[1].weight()
    obstacle_positions = []
    obstacle_radii = []
    for obst in obstacles:
        obstacle_positions.append(obst.position())
        obstacle_radii.append(np.array(obst.radius()))

    sub_goal_0_quaternion = quaternionic.array(goal.subGoals()[1].angle())
    sub_goal_0_rotation_matrix = sub_goal_0_quaternion.to_rotation_matrix

    for i in range(n_steps):

        action = planner.compute_action(
            q=ob["joint_state"]["position"],
            qdot=ob["joint_state"]["velocity"],
            x_goal_0=sub_goal_0_position,
            weight_goal_0=sub_goal_0_weight,
            x_goal_1=sub_goal_1_position,
            weight_goal_1=sub_goal_1_weight,
            x_obsts = obstacle_positions,
            radius_obsts = obstacle_radii,
            radius_body_panda_link1=0.1,
            radius_body_panda_link4=0.1,
            radius_body_panda_link6=0.15,
            radius_body_panda_hand=0.1,
            angle_goal_1=np.array(sub_goal_0_rotation_matrix),
        )
        ob, *_ = env.step(action)

    return {}


if __name__ == "__main__":
    res = run_panda_ring_example(n_steps=10000, serialize = True)
