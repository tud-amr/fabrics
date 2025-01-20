import logging
import time
import os
import gymnasium as gym
import numpy as np

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor
from urdfenvs.sensors.lidar import Lidar

from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.goals.goal_composition import GoalComposition

from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

from point_robot_fsd import get_obstacles_fsd, get_goal_fsd

logging.basicConfig(level=logging.ERROR)

NUMBER_OF_RAYS = 20

def get_goal_sensors():
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
    return goal

def get_obstacles_sensors():
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
    return [obst1, obst2, obst3]

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
    env: UrdfEnv  = UrdfEnv(
        dt=0.01, robots=robots, render=render
    ).unwrapped

    # Set the initial position and velocity of the point mass.
    pos0 = np.array([-2.0, 0.5, 0.0])
    vel0 = np.array([0.0, 0.0, 0.0])
    full_sensor = FullSensor(
        goal_mask=["weight", "position"],
        obstacle_mask=["position", "size"],
        variance=0.0,
    )
    lidar = Lidar(4, nb_rays=NUMBER_OF_RAYS, raw_data=False)
    # Definition of the obstacle.
    obstacles = get_obstacles_fsd()
    # Definition of the goal.
    goal = get_goal_fsd()
    env.reset(pos=pos0, vel=vel0)
    env.add_sensor(full_sensor, [0])
    env.add_sensor(lidar, robot_ids=[0])
    env.add_goal(goal.sub_goals()[0])
    for obst in obstacles:
        env.add_obstacle(obst)
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
    collision_geometry = "-0.2 / (x ** 2) * xdot ** 2"
    collision_finsler = "0.1 / (x ** 2) * (1 - ca.heaviside(xdot))* xdot**2"
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    with open(absolute_path + "/point_robot.urdf", "r") as file:
        urdf = file.read()
    forward_kinematics = GenericURDFFk(
        urdf,
        root_link="world",
        end_links="base_link",
    )
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        forward_kinematics,
        collision_geometry=collision_geometry,
        collision_finsler=collision_finsler
    )
    collision_links = ['base_link']
    planner.set_components(
        goal=goal,
        collision_links=collision_links,
        number_obstacles=NUMBER_OF_RAYS,
    )
    planner.concretize()
    return planner

def set_lidar_runtime_arguments(robot_position, lidar_observation: np.ndarray) -> dict:
    lidar_runtime_arguments = {}
    relative_positions = np.concatenate((np.reshape(lidar_observation, (NUMBER_OF_RAYS, 2)), np.zeros((NUMBER_OF_RAYS, 1))), axis=1)
    absolute_ray_end_positions = relative_positions + np.repeat(robot_position[np.newaxis, :], NUMBER_OF_RAYS, axis=0)
    for ray_id in range(NUMBER_OF_RAYS):
        lidar_runtime_arguments[f'x_obst_{ray_id}'] = absolute_ray_end_positions[ray_id]
        lidar_runtime_arguments[f'radius_obst_{ray_id}'] = 0.1
    return lidar_runtime_arguments


def run_point_robot_sensor(n_steps=10000, render=True):
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
    env.reconfigure_camera(5, 0, 270.1, [0, 0, 0])


    for _ in range(n_steps):
        t0 = time.perf_counter()
        ob_robot = ob['robot_0']
        lidar_runtime_arguments = set_lidar_runtime_arguments(ob_robot['joint_state']['position'], ob_robot['LidarSensor'])
        action = planner.compute_action(
            q=ob_robot["joint_state"]["position"],
            qdot=ob_robot["joint_state"]["velocity"],
            x_goal_0=ob_robot['FullSensor']['goals'][2]['position'][0:2],
            weight_goal_0=ob_robot['FullSensor']['goals'][2]['weight'],
            radius_body_base_link=np.array([0.35]),
            **lidar_runtime_arguments,
        )
        ob, *_, = env.step(action)
        t1 = time.perf_counter()
        print(t1-t0)
    return {}

if __name__ == "__main__":
    res = run_point_robot_sensor(n_steps=100000, render=True)
