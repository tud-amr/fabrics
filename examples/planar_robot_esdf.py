import os
import sys
from copy import deepcopy
import math
import logging
import gymnasium as gym
import numpy as np

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor

from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle

import pybullet as p
from scipy import ndimage

from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

logging.basicConfig(level=logging.INFO)

def edf(pos, proj_rgb) -> tuple:
    # to binary image, obstacle are red
    proj_r = proj_rgb[:, :, 1]
    proj_bin = ((1 - proj_r) > 0.9)

    pixel_to_meter_ratio = 0.1
    offset = np.array([6.5, 5])
    direction_multiplier = np.array([1, -1])

    dist_map = ndimage.distance_transform_edt(1 - proj_bin)
    dist_map = dist_map * pixel_to_meter_ratio  # /100 pixels * 20 in meters

    # convert pos to pixels
    pos_p = np.rint((direction_multiplier * pos + offset) / pixel_to_meter_ratio)
    if math.isnan(pos_p[0]):
        kkk_stop = 1
    pos_p = [int(pos_pi) for pos_pi in pos_p]

    # index in map
    dist_pos = dist_map[int(pos_p[1]), int(pos_p[0])]

    gradient_map = np.gradient(dist_map)
    gradient_x = gradient_map[1] / pixel_to_meter_ratio
    gradient_y = -gradient_map[0] / pixel_to_meter_ratio
    grad_x_pos = gradient_x[pos_p[1], pos_p[0]]
    grad_y_pos = gradient_y[pos_p[1], pos_p[0]]
    return (dist_pos, grad_x_pos, grad_y_pos)  # dist_map


def get_top_view_image(save=False, load_only=False):
    try:
        rgb_file = os.path.dirname(os.path.abspath(__file__)) + "/proj_rgb_planar_arm.npy"
        depth_file = os.path.dirname(os.path.abspath(__file__)) + "/proj_depth_planar_arm.npy"
        proj_rgb = np.load(rgb_file)
        proj_depth = np.load(depth_file)
    except FileNotFoundError as e:
        if load_only:
            raise(e)
        width_res = 130
        height_res = 100
        img = p.getCameraImage(width_res, height_res, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        proj_rgb = np.reshape(img[2], (height_res, width_res, 4)) * 1. / 255.
        proj_depth = img[3]
        if save:
            np.save('proj_rgb_planar_arm', proj_rgb)
            np.save('proj_depth_planar_arm', proj_depth)
    return proj_rgb, proj_depth

def initalize_environment(render=True):
    """
    Initializes the simulation environment.

    Adds obstacles and goal visualizaion to the environment based and
    steps the simulation once.
    """
    urdf_file = os.path.dirname(os.path.abspath(__file__)) + "/planar_urdf_2_joints.urdf"
    robots = [
        GenericUrdfReacher(urdf=urdf_file, mode="acc"),
    ]
    env: UrdfEnv  = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    full_sensor = FullSensor(
            goal_mask=["position", "weight"],
            obstacle_mask=["position", "size"],
            variance=0.0,
    )
    # Definition of the obstacle.
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [0.0, -5.5, 1.0], "radius": 0.3},
    }
    obst1 = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [-0.0, 1.2, 2], "radius": 0.3},
    }
    obst2 = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
    # Definition of the goal.
    goal_dict = {
        "subgoal0": {
            "weight": 3.0,
            "is_primary_goal": True,
            "indices": [1, 2],
            "parent_link": "panda_link0",
            "child_link": "panda_link4",
            "desired_position": [1.5, 0.8],
            "epsilon": 0.05,
            "type": "staticSubGoal",
        },
    }
    visualize_goal_dict = deepcopy(goal_dict)
    visualize_goal_dict['subgoal0']['indices'] = [0] + goal_dict['subgoal0']['indices']
    visualize_goal_dict['subgoal0']['desired_position'] = [0.0] + goal_dict['subgoal0']['desired_position']
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    vis_goal = GoalComposition(name="goal", content_dict=visualize_goal_dict)
    obstacles = (obst1, obst2) #, obst2)
    vel0 = np.array([-0.1, -0.1])
    env.reset(vel=vel0)
    env.add_sensor(full_sensor, [0])
    for obst in obstacles:
        env.add_obstacle(obst)
    for sub_goal in vis_goal.sub_goals():
        env.add_goal(sub_goal)
    env.set_spaces()
    return (env, goal)


def set_planner(goal: GoalComposition, degrees_of_freedom: int = 2):
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
    collision_geometry = "-1.0 / (x ** 2) * xdot ** 2"
    collision_finsler = "0.2/(x ** 2) * (1 - ca.heaviside(xdot))* xdot**2"
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    with open(absolute_path + "/planar_urdf_2_joints.urdf", "r") as file:
        urdf = file.read()
    forward_kinematics = GenericURDFFk(
        urdf,
        rootLink="panda_link0",
        end_link="panda_link4",
    )
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        forward_kinematics,
        collision_geometry=collision_geometry,
        collision_finsler=collision_finsler
    )
    collision_links = ['panda_link4']
    planner.set_components(
        collision_links_esdf=collision_links,
        goal=goal,
    )
    planner.concretize()
    return planner


def run_planar_robot_esdf(n_steps=5000, render=True):
    (env, goal) = initalize_environment(render)
    planner = set_planner(goal)
    action = -np.zeros(env.n())
    ob, *_ = env.step(action)
    if render:
        p.resetDebugVisualizerCamera(5, 90, 0, [0, 0, 0])
        input("Make sure that the pybullet window is in default window size. Then press any key.")
        proj_rgb, proj_depth = get_top_view_image(save=True)

    for _ in range(n_steps):
        ob_robot = ob['robot_0']
        pos_joints = ob_robot['joint_state']['position'][0:2]

        pos_link4_fun = planner._forward_kinematics._fks["panda_link4"]
        pos_link4 = pos_link4_fun(pos_joints)[1:3, 3]  # only return (y, z)
        proj_rgb, _ = get_top_view_image(save=False, load_only=True)
        edf_phi, edf_gradient_x, edf_gradient_y = edf(
            pos_link4,
            proj_rgb
        )
        edf_gradient_phi_x = np.array([0.0, edf_gradient_x, edf_gradient_y])
        action = planner.compute_action(
            q=ob_robot["joint_state"]["position"],
            qdot=ob_robot["joint_state"]["velocity"],
            x_goal_0=goal.sub_goals()[0].position(),
            weight_goal_0=goal.sub_goals()[0].weight(),
            esdf_phi_panda_link4=edf_phi,
            esdf_J_panda_link4=edf_gradient_phi_x,
            esdf_Jdot_panda_link4=np.zeros(2),
            radius_body_panda_link4=0.1,
        )

        ob, *_ = env.step(action)
    env.close()
    return {}


if __name__ == "__main__":
    res = run_planar_robot_esdf(n_steps=5000, render=bool(sys.argv[1]))
