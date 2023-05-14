import logging
import os

import gym
import numpy as np
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.goals.goal_composition import GoalComposition
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner
import pybullet as p
from scipy import ndimage

logging.basicConfig(level=logging.ERROR)


def edf(pos, proj_rgb) -> tuple:
    #to binary image, obstacle are red
    proj_r = proj_rgb[:, :, 1]
    proj_bin = ((1-proj_r) > 0.9)

    pixel_to_meter_ratio = 0.1
    offset = np.array([6.5, 5])
    direction_multiplier = np.array([1, -1])
    
    dist_map = ndimage.distance_transform_edt(1-proj_bin)
    dist_map = dist_map * pixel_to_meter_ratio # /100 pixels * 20 in meters

    # convert pos to pixels
    pos_p = np.rint((direction_multiplier * pos + offset)/pixel_to_meter_ratio)
    pos_p = [int(pos_pi) for pos_pi in pos_p]

    dist_pos = dist_map[int(pos_p[1]), int(pos_p[0])]

    gradient_map = np.gradient(dist_map)
    gradient_x = gradient_map[1]/pixel_to_meter_ratio
    gradient_y = -gradient_map[0]/pixel_to_meter_ratio
    grad_x_pos = gradient_x[pos_p[1], pos_p[0]]
    grad_y_pos = gradient_y[pos_p[1], pos_p[0]]
    return (dist_pos, grad_x_pos, grad_y_pos) #dist_map

def get_top_view_image(save=False, load_only=False):
    try:
        proj_rgb = np.load("proj_rgb_point_robot.npy")
        proj_depth = np.load("proj_depth_point_robot.npy")
    except FileNotFoundError as e:
        if load_only:
            raise(e)
        width_res = 130
        height_res = 100
        img = p.getCameraImage(width_res, height_res, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        proj_rgb = np.reshape(img[2], (height_res, width_res, 4)) * 1. / 255.
        proj_depth = img[3]
        if save:
            np.save('proj_rgb_point_robot', proj_rgb)
            np.save('proj_depth_point_robot', proj_depth)
    return proj_rgb, proj_depth

def test_edf_evaluation(proj_rgb):
    edf_test, *_ = edf([2.0, 0.0], proj_rgb)
    assert edf_test == 0.0
    _, edf_gradient_x, edf_gradient_y = edf(
        [4.0, 2.0],
        proj_rgb
    )
    assert edf_gradient_x >= 0
    assert edf_gradient_y >= 0
    _, edf_gradient_x, edf_gradient_y = edf(
        [4.0, -2.0],
        proj_rgb
    )
    assert edf_gradient_x >= 0
    assert edf_gradient_y <= 0
    _, edf_gradient_x, edf_gradient_y = edf(
        [-4.0, -2.0],
        proj_rgb
    )
    assert edf_gradient_x <= 0
    assert edf_gradient_y <= 0
    _, edf_gradient_x, edf_gradient_y = edf(
        [-4.0, 2.0],
        proj_rgb
    )
    assert edf_gradient_x <= 0
    assert edf_gradient_y >= 0

def initalize_environment(render):
    """
    Initializes the simulation environment.

    Adds an obstacle and goal visualizaion to the environment and
    steps the simulation once.
    
    Params
    ----------
    render
        Boolean toggle to set rendering on (True) or off (False).
    """
    robots = [
        GenericUrdfReacher(urdf="point_robot.urdf", mode="acc"),
    ]
    env: UrdfEnv  = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    # Set the initial position and velocity of the point mass.
    pos0 = np.array([-2.0, 0.5, 0.0])
    vel0 = np.array([0.0, 0.0, 0.0])
    full_sensor = FullSensor(
            goal_mask=["position", "weight"],
            obstacle_mask=[],
            variance=0.0
    )
    # Definition of the obstacle.
    static_obst_dict_1 = {
            "type": "sphere",
            "geometry": {"position": [2.0, 1.2, 0.0], "radius": 1.0},
    }
    static_obst_dict_2 = {
            "type": "sphere",
            "geometry": {"position": [0.5, -0.8, 0.0], "radius": 0.4},
    }
    static_obst_dict_3 = {
            "type": "sphere",
            "geometry": {"position": [0.0, 4.0, 0.0], "radius": 1.5},
    }
    obst1 = SphereObstacle(name="staticObst1", content_dict=static_obst_dict_1)
    obst2 = SphereObstacle(name="staticObst2", content_dict=static_obst_dict_2)
    obst3 = SphereObstacle(name="staticObst3", content_dict=static_obst_dict_3)
    # Definition of the goal.
    goal_dict = {
            "subgoal0": {
                "weight": 0.5,
                "is_primary_goal": True,
                "indices": [0, 1],
                "parent_link" : 'world',
                "child_link" : 'base_link',
                "desired_position": [3.5, 0.5],
                "epsilon" : 0.1,
                "type": "staticSubGoal"
            }
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    env.reset(pos=pos0, vel=vel0)
    env.add_sensor(full_sensor, [0])
    env.add_goal(goal.sub_goals()[0])
    env.add_obstacle(obst1)
    env.add_obstacle(obst2)
    env.add_obstacle(obst3)
    env.set_spaces()
    return (env, goal)


def set_planner(goal: GoalComposition):
    """
    Initializes the fabric planner for the point robot.

    This function defines the forward kinematics for collision avoidance,
    and goal reaching. These components are fed into the fabrics planner.

    In the top section of this function, an example for optional reconfiguration
    can be found. Commented by default.

    Params
    ----------
    goal: GoalComposition
        The goal to the motion planning problem.
    """
    degrees_of_freedom = 3
    robot_type = "xyz"
    # Optional reconfiguration of the planner with collision_geometry/finsler, remove for defaults.
    collision_geometry = "-0.8 / (x ** 2) * xdot ** 2"
    collision_finsler = "0.5/(x ** 2) * (1 - ca.heaviside(xdot))* xdot**2"
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    with open(absolute_path + "/point_robot.urdf", "r") as file:
        urdf = file.read()
    planner = ParameterizedFabricPlanner(
            degrees_of_freedom,
            robot_type,
            urdf=urdf,
            root_link='world',
            end_link='base_link',
            collision_geometry=collision_geometry,
            collision_finsler=collision_finsler
    )
    collision_links_esdf = ['base_link']
    planner.set_components(
        goal=goal,
        collision_links_esdf=collision_links_esdf,
    )
    planner.concretize()
    return planner


def run_point_robot_esdf(n_steps=10000, render=True):
    """
    Set the gym environment, the planner and run point robot example.
    The initial zero action step is needed to initialize the sensor in the
    urdf environment.

    Params
    ----------
    n_steps
        Total number of simulation steps.
    render
        Boolean toggle to set rendering on (True) or off (False).
    """
    (env, goal) = initalize_environment(render)
    planner = set_planner(goal)

    action = np.array([0.0, 0.0, 0.0])
    ob, *_ = env.step(action)
    if render:
        p.resetDebugVisualizerCamera(5, 0, 270.1, [0, 0, 0])
        input("Make sure that the pybullet window is in default window size. Then press any key.")
        proj_rgb, proj_depth = get_top_view_image(save=True)

    if logging.root.level <= 10:
        plt.subplot(1, 2, 1)
        plt.imshow(proj_rgb)
        plt.subplot(1, 2, 2)
        plt.imshow(proj_depth)
        plt.show()
        test_edf_evaluation(proj_rgb)

    for _ in range(n_steps):
        ob_robot = ob['robot_0']
        proj_rgb, proj_depth = get_top_view_image(load_only=True, save=False)

        edf_phi, edf_gradient_x, edf_gradient_y = edf(
            ob_robot['joint_state']['position'][0:2],
            proj_rgb
        )

        
        edf_gradient = np.array([edf_gradient_x, edf_gradient_y, 0.0])        
        #print(edf_gradient)
        action = planner.compute_action(
            q=ob_robot["joint_state"]["position"],
            qdot=ob_robot["joint_state"]["velocity"],
            x_goal_0=ob_robot['FullSensor']['goals'][2]['position'][0:2],
            weight_goal_0=ob_robot['FullSensor']['goals'][2]['weight'],
            esdf_phi_base_link=edf_phi,
            esdf_J_base_link=edf_gradient,
            esdf_Jdot_base_link=np.zeros(3),
            radius_body_base_link=np.array([0.4])
        )
        ob, *_, = env.step(action)
    env.close()
    return {}

if __name__ == "__main__":
    res = run_point_robot_esdf(n_steps=100000, render=True)
