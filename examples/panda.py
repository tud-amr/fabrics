import os
import shutil
import sys
import gymnasium as gym
import numpy as np
import yaml

from robotmodels.utils.robotmodel import RobotModel, LocalRobotModel

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

from urdfenvs.sensors.full_sensor import FullSensor

from urdfenvs.generic_mujoco.generic_mujoco_robot import GenericMujocoRobot
from urdfenvs.generic_mujoco.generic_mujoco_env import GenericMujocoEnv

from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.obstacles.box_obstacle import BoxObstacle

from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

CONFIG_FILE = "panda_config.yaml"
with open(CONFIG_FILE, 'r') as config_file:
    config = yaml.safe_load(config_file)
    CONFIG_PROBLEM = config['problem']
    CONFIG_FABRICS = config['fabrics']

ROBOTTYPE = 'panda'
ROBOTMODEL = 'panda_without_gripper'



def initalize_environment(render=True):
    """
    Initializes the simulation environment.

    Adds obstacles and goal visualizaion to the environment based and
    steps the simulation once.
    """
    full_sensor = FullSensor(
            goal_mask=["position", "weight"],
            obstacle_mask=['position', 'size'],
            variance=0.0,
            physics_engine_name='mujoco',
    )
    
    if not os.path.exists(f'{ROBOTTYPE}_local'):
        robot_model_original = RobotModel(ROBOTTYPE, ROBOTMODEL)
        robot_model_original.copy_model(os.path.join(os.getcwd(), f'{ROBOTTYPE}_local'))
    robot_model = LocalRobotModel(f'{ROBOTTYPE}_local', ROBOTMODEL)

    xml_file = robot_model.get_xml_path()
    robots  = [
        GenericMujocoRobot(xml_file=xml_file, mode="vel"),
    ]
    # Definition of the obstacle.
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [0.3, -0.3, 0.65], "radius": 0.1},
    }
    obst1 = SphereObstacle(name="obstacle_0", content_dict=static_obst_dict)
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [-0.3, 0.0, 0.5], "radius": 0.1},
    }
    obst2 = SphereObstacle(name="obstacle_1", content_dict=static_obst_dict)
    static_obst_dict = {
        "type": "box",
        "geometry": {"position": [-0.0, -0.6, 0.45],
                     "length": 1.1,
                     "width": 0.5,
                     "height": 0.1,
                     },
    }
    obst3 = BoxObstacle(name="staticObst", content_dict=static_obst_dict)

    goal = GoalComposition(name="goal", content_dict=CONFIG_PROBLEM['goal']['goal_definition'])
    obstacles = [obst1, obst2, obst3]
    env = GenericMujocoEnv(
        robots=robots,
        obstacles=obstacles,
        goals=goal.sub_goals(),
        sensors=[full_sensor],
        render=render,
        enforce_real_time=True,
    )
    env.reset()
    return env


def set_planner(degrees_of_freedom: int = 7):
    """
    Initializes the fabric planner for the panda robot.

    Reparameterization is done in the panda_config.yaml file.

    Params
    ----------
    degrees_of_freedom: int
        Degrees of freedom of the robot (default = 7)
    """
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    with open(absolute_path + "/panda_for_fk.urdf", "r", encoding="utf-8") as file:
        urdf = file.read()
    forward_kinematics = GenericURDFFk(
        urdf,
        root_link="panda_link0",
        end_links=["panda_link9"],
    )
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        forward_kinematics,
    )
    planner.load_fabrics_configuration(CONFIG_FABRICS)
    planner.load_problem_configuration(CONFIG_PROBLEM)
    planner.concretize(mode='vel', time_step=0.05)
    return planner


def run_panda_example(n_steps=5000, render=True):
    env = initalize_environment(render)
    planner = set_planner()
    # planner.export_as_c("planner.c")
    action = np.zeros(7)
    ob, *_ = env.step(action)


    for _ in range(n_steps):
        ob_robot = ob['robot_0']
        all_arguments = dict(
            q=ob_robot["joint_state"]["position"][0:7],
            qdot=ob_robot["joint_state"]["velocity"][0:7],
            x_goal_0=ob_robot['FullSensor']['goals'][0]['position'],
            weight_goal_0=ob_robot['FullSensor']['goals'][0]['weight'],
            x_goal_1=ob_robot['FullSensor']['goals'][1]['position'],
            weight_goal_1=ob_robot['FullSensor']['goals'][1]['weight'],
            x_obst_0=ob_robot['FullSensor']['obstacles'][0]['position'],
            radius_obst_0=ob_robot['FullSensor']['obstacles'][0]['size'],
            x_obst_1=ob_robot['FullSensor']['obstacles'][1]['position'],
            radius_obst_1=ob_robot['FullSensor']['obstacles'][1]['size'],
            x_obst_2=ob_robot['FullSensor']['obstacles'][2]['position'],
            sizes_obst_2=ob_robot['FullSensor']['obstacles'][2]['size'],
            constraint_0=np.array([0, 0, 1, 0.0]),
            #**arguments,
        )
        action = planner.compute_action(**all_arguments)
        ob, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(info)
    env.close()
    return {}


if __name__ == "__main__":
    render = False
    if len(sys.argv) > 1:
        render = bool(int(sys.argv[1]))
    res = run_panda_example(render=render, n_steps=1000)
