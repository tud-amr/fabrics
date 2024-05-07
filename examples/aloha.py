import sys
import time
import gymnasium as gym
import numpy as np
import yaml
import matplotlib.pyplot as plt

from robotmodels.utils.robotmodel import RobotModel

from forwardkinematics.xmlFks.generic_xml_fk import GenericXMLFk

from urdfenvs.sensors.full_sensor import FullSensor

from urdfenvs.generic_mujoco.generic_mujoco_robot import GenericMujocoRobot
from urdfenvs.generic_mujoco.generic_mujoco_env import GenericMujocoEnv

from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.obstacles.box_obstacle import BoxObstacle

from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

MODE = 'jointspace' # eespace or jointspace
if MODE == 'eespace':
    CONFIG_FILE = "aloha_config.yaml"
    GOALS = [
        [0.3, -0.36, 0.36],
        [0.3, -0.36, 0.36],
    ]
    WEIGHT=50
else:
    CONFIG_FILE = "aloha_joint_space_config.yaml"
    GOALS= [
        [ 0.7 , -0.96,  1.16,  0.  , -0.3 ,  0.],
        [ 0.  , -0.96,  1.16,  0.  , -0.3 ,  0.],
    ]
    WEIGHT=50
with open(CONFIG_FILE, 'r') as config_file:
    config = yaml.safe_load(config_file)
    CONFIG_PROBLEM = config['problem']
    CONFIG_FABRICS = config['fabrics']

ROBOTTYPE = 'aloha'
ROBOTMODEL = 'aloha_scene'



def initalize_environment(robot_model: RobotModel, render: bool = True):
    """
    Initializes the simulation environment.

    Adds obstacles and goal visualizaion to the environment based and
    steps the simulation once.
    """

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
        obstacles=[],
        goals=[],
        sensors=[],
        render=render,
        enforce_real_time=True,
    )
    env.reset(pos=robot_model.home_cfg())
    return env


def set_planner(robot_model: RobotModel):
    """
    Initializes the fabric planner for the aloha setup.

    Reparameterization is done in the aloha_config.yaml file.

    Params
    ----------
    degrees_of_freedom: int
        Degrees of freedom of the robot (default = 7)
    """
    robot_model = RobotModel(ROBOTTYPE, 'aloha')
    with open(robot_model.get_xml_path(), "r", encoding="utf-8") as file:
        xml = file.read()
    forward_kinematics = GenericXMLFk(
        xml,
        root_link="left_base_link",
        end_links=["left_gripper_link"],
    )
    planner = ParameterizedFabricPlanner(
        6,
        forward_kinematics,
    )
    planner.load_fabrics_configuration(CONFIG_FABRICS)
    planner.load_problem_configuration(CONFIG_PROBLEM)
    planner.concretize()
    return planner


def run_panda_example(n_steps=5000, render=True):
    robot_model = RobotModel(ROBOTTYPE, ROBOTMODEL)
    env = initalize_environment(robot_model, render)
    planner = set_planner(robot_model)
    # planner.export_as_c("planner.c")
    pos0 = robot_model.home_cfg()


    action = np.concatenate((pos0[0:7], pos0[8:15]))
    ob, *_ = env.step(action)
    for i in range(1):
        ob, *_ = env.step(action)
        q = ob['robot_0']['joint_state']['position'][0:6]
        qdot = ob['robot_0']['joint_state']['velocity'][0:6]


    state_indices = [
        [0, 1, 2, 3, 4, 5],
        [8, 9, 10, 11, 12, 13],
    ]
    action_indices = [
        [0, 1, 2, 3, 4, 5],
        [7, 8, 9, 10, 11, 12],
    ]
    qs = np.zeros((n_steps, 6))
    ts = np.zeros((n_steps, 1))
    for i in range(n_steps):
        ts[i] = env.t
        ob_robot = ob['robot_0']
        for robot_id in range(1):
            t0 = time.perf_counter()
            q=ob_robot["joint_state"]["position"][state_indices[robot_id]]
            qs[i] = q
            qdot=ob_robot["joint_state"]["velocity"][state_indices[robot_id]]
            all_arguments = dict(
                q=q,
                qdot=qdot,
                x_goal_0=GOALS[robot_id],
                weight_goal_0=WEIGHT,
            )
            qddot_des = planner.compute_action(**all_arguments)
            qdot_des = qdot + env.dt * qddot_des
            q_des = q + env.dt * (0.5 * qdot_des + 0.5 * qdot)
            if i % 10 == -1:
                if MODE == 'eespace':
                    fk_ee = planner._forward_kinematics.numpy(q, 'left_gripper_link', position_only=True)
                    fk_base = planner._forward_kinematics.numpy(q, 'left_base_link', position_only=True)
                    fk = fk_ee-fk_base
                    error = np.round(fk - GOALS[robot_id], 3)
                else:
                    error = np.round((q - GOALS[robot_id]) * 180/np.pi, 3)
                print(f"error[{robot_id}] = {error}")

            t1 = time.perf_counter()
            print(f"Time for planning: {(t1-t0)*1000} ms")

            action[action_indices[robot_id]] = q_des
            #action[action_indices[robot_id]] = GOALS[robot_id]
        ob, reward, terminated, truncated, info = env.step(action)
    env.close()
    #np.save('pid_limit', qs)
    pid_qs = np.load('pid_limit.npy')
    plt.plot(ts, qs[:,0], label='fabric')
    plt.plot(ts, pid_qs[:,0], label='p')
    plt.legend()
    plt.show()
    return {}


if __name__ == "__main__":
    render = False
    if len(sys.argv) > 1:
        render = bool(int(sys.argv[1]))
    res = run_panda_example(render=render, n_steps=1000)
