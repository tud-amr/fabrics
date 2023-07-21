import gymnasium as gym
from pynput import keyboard
from typing import List
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from mpscenes.obstacles.collision_obstacle import CollisionObstacle
from mpscenes.obstacles.box_obstacle import BoxObstacle
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor
import os
import quaternionic

from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle

import numpy as np
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

URDF_FILE = os.path.dirname(os.path.abspath(__file__)) + "/panda_with_finger.urdf"
GRIPPER_ACTION = 0
GO_UP = 0


def key_press_callback(e):
    # Handle the key press event here
    global GRIPPER_ACTION
    global GO_UP
    try:
        if e.char == 'c':
            print("Closing gripper")
            GRIPPER_ACTION = 1

        elif e.char == 'o':
            GRIPPER_ACTION = 0
            print("Opening gripper")
        elif e.char == 'u':
            GO_UP = 1
        elif e.char == 'd':
            GO_UP = 0
    except AttributeError:
        pass




def create_scene() -> List[CollisionObstacle]:
    table_dict = {
        "type": "box",
        "geometry": {
            "position": [0.5, -0.0, 0.1],
            "length": 0.6,
            "height": 0.2,
            "width": 1.0,
        },
    }
    obst_1 = BoxObstacle(name="table", content_dict=table_dict)
    object_dict = {
        "type": "box",
        "movable": True,
        "geometry": {
            "position": [0.6, -0.0, 0.3],
            "length": 0.05,
            "height": 0.05,
            "width": 0.05,
        },
    }
    object_obstacle = BoxObstacle(name="table", content_dict=object_dict)
    return [obst_1, object_obstacle]

def create_dummy_goal() -> GoalComposition:
    goal_orientation = [-0.366, 0.0, 0.0, 0.3305]
    whole_position = [0.1, 0.6, 0.8]
    goal_dict = {
        "subgoal0": {
            "weight": 1.0,
            "is_primary_goal": True,
            "indices": [0, 1, 2],
            "parent_link": "panda_link0",
            "child_link": "panda_hand",
            "desired_position": whole_position,
            "epsilon": 0.05,
            "type": "staticSubGoal",
        },
        "subgoal1": {
            "weight": 3.0,
            "is_primary_goal": False,
            "indices": [0, 1, 2],
            "parent_link": "panda_link7",
            "child_link": "panda_hand",
            "desired_position": [0.107, 0.0, 0.0],
            "angle": goal_orientation,
            "epsilon": 0.05,
            "type": "staticSubGoal",
        },
        "subgoal2": {
            "weight": 3.0,
            "is_primary_goal": False,
            "indices": [6],
            "desired_position": [np.pi/4],
            "epsilon": 0.05,
            "type": "staticJointSpaceSubGoal",
        }
    }
    return GoalComposition(name="goal", content_dict=goal_dict)


def initalize_environment(render=True):
    """
    Initializes the simulation environment.

    Adds obstacles and goal visualizaion to the environment based and
    steps the simulation once.
    """
    robots = [
        GenericUrdfReacher(urdf=URDF_FILE, mode="vel"),
    ]
    env: UrdfEnv  = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    full_sensor = FullSensor(
            goal_mask=[],
            obstacle_mask=["position", "size"],
            variance=0.0,
    )
    # Definition of the obstacle.
    obstacles = create_scene()
    env.reset()
    env.add_sensor(full_sensor, [0])
    for obst in obstacles:
        env.add_obstacle(obst)
    env.set_spaces()
    return env


def set_planner(degrees_of_freedom: int = 7):
    """
    Initializes the fabric planner for the panda robot.

    This function defines the forward kinematics for collision avoidance,
    and goal reaching. These components are fed into the fabrics planner.

    In the top section of this function, an example for optional reconfiguration
    can be found. Commented by default.

    Params
    ----------
    degrees_of_freedom: int
        Degrees of freedom of the robot (default = 7)
    """
    goal = create_dummy_goal()
    with open(URDF_FILE, "r", encoding="utf-8") as file:
        urdf = file.read()
    forward_kinematics = GenericURDFFk(
        urdf,
        rootLink="panda_link0",
        end_link="panda_leftfinger",
    )
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        forward_kinematics,
    )
    collision_links = ["panda_hand", "panda_link3", "panda_link4"]
    panda_limits = [
            [-2.8973, 2.8973],
            [-1.7628, 1.7628],
            [-2.8973, 2.8973],
            [-3.0718, -0.0698],
            [-2.8973, 2.8973],
            [-0.0175, 3.7525],
            [-2.8973, 2.8973]
        ]
    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        collision_links=collision_links,
        goal=goal,
        number_obstacles=0,
        limits=panda_limits,
    )
    planner.concretize(mode='vel', time_step=0.01)
    return planner


def run_panda_example(n_steps=5000, render=True):
    env = initalize_environment(render)
    planner = set_planner()
    action = np.zeros(9)
    goal_orientation = [0.0, 0.707, 0.707, 0.0]
    yaw = -np.pi/4
    rotation_matrix_2 = np.array([
        [np.cos(yaw), -np.sin(yaw), 0], 
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
        ])
    rotation_matrix = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    ob, *_ = env.step(action)
    listener = keyboard.Listener(on_press=key_press_callback)
    listener.start()

    print("Control the gripper using 'o' and 'c' for opening and closing the gripper")
    print("Press 'u' to go to a home pose after grasping")
    print("Press 'd' to go to the object again.")

    env.reconfigure_camera(1.4, 85.0, -25.0, (0.0, 0.0, 0.0))


    for _ in range(n_steps):
        ob_robot = ob['robot_0']
        q=ob_robot["joint_state"]["position"][0:7]
        qdot=ob_robot["joint_state"]["velocity"][0:7]
        object_position = ob_robot['FullSensor']['obstacles'][3]['position']
        object_position[2] += 0.1
        if GO_UP:
            object_position = [0.5, 0.3, 0.8]
        action[0:7] = planner.compute_action(
            q=q,
            qdot=qdot,
            x_goal_0 = np.array(object_position),
            weight_goal_0=0.7,
            angle_goal_1=rotation_matrix,
            x_goal_1 = np.array([0.107, 0.0, 0.0]),
            weight_goal_1=6.0,
            x_goal_2 = np.array([np.pi/4]),
            weight_goal_2=6.0,
            radius_body_panda_link3=np.array([0.02]),
            radius_body_panda_link4=np.array([0.02]),
            radius_body_panda_hand=np.array([0.02]),
        )
        distance_to_full_opening = np.linalg.norm(q[-2:] - np.array([0.04, 0.04]))
        distance_to_full_closing = np.linalg.norm(q[-2:] - np.array([0.00, 0.00]))
        if GRIPPER_ACTION == 0:# and distance_to_full_opening > 0.01:

            action[7:] = np.ones(2) * 0.05
        elif GRIPPER_ACTION == 1:# and distance_to_full_closing > 0.01:
            action[7:] = np.ones(2) * -0.05
        else:
            action[7:] = np.zeros(2)
        ob, *_ = env.step(action)
    listener.stop()
    listener.join()
    return {}


if __name__ == "__main__":
    res = run_panda_example(n_steps=5000)
