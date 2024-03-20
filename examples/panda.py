import os
import sys
import gymnasium as gym
import numpy as np
import yaml

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor

from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.obstacles.box_obstacle import BoxObstacle

from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

CONFIG_FILE = "panda_config.yaml"
with open(CONFIG_FILE, 'r') as config_file:
    config = yaml.safe_load(config_file)
    CONFIG_PROBLEM = config['problem']
    CONFIG_FABRICS = config['fabrics']



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
    full_sensor = FullSensor(
            goal_mask=["position", "weight"],
            obstacle_mask=['position', 'size'],
            variance=0.0
    )
    # Definition of the obstacle.
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [0.5, -0.3, 0.3], "radius": 0.1},
    }
    obst1 = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [-0.7, 0.0, 0.5], "radius": 0.1},
    }
    obst2 = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
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
    obstacles = (obst1, obst2, obst3)
    vel0 = np.array([-0.0, 0, 0, 0, 0, 0, 0])
    env.reset(vel=vel0)
    env.add_sensor(full_sensor, [0])
    for obst in obstacles:
        env.add_obstacle(obst)
    for sub_goal in goal.sub_goals():
        env.add_goal(sub_goal)
    env.set_spaces()
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
        rootLink="panda_link0",
        end_link="panda_link9",
    )
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        forward_kinematics,
    )
    planner.load_fabrics_configuration(CONFIG_FABRICS)
    planner.load_problem_configuration(CONFIG_PROBLEM)
    planner.concretize()
    return planner


def run_panda_example(n_steps=5000, render=True):
    env = initalize_environment(render)
    planner = set_planner()
    # planner.export_as_c("planner.c")
    action = np.zeros(7)
    ob, *_ = env.step(action)
    body_links={1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 7: 0.1}
    # Passing arguments for the robot representation is possible but not needed.
    # If nothing is specified, the defaults from the config file are used.
    arguments = {}
    for body_link, radius in body_links.items():
        env.add_collision_link(0, body_link, shape_type='sphere', size=[radius])
        arguments[f'radius_panda_link{body_link}'] = radius
    arguments['length_panda_link5'] = 0.03
    arguments['radius_panda_link5'] = 0.01


    for _ in range(n_steps):
        ob_robot = ob['robot_0']
        all_arguments = dict(
            q=ob_robot["joint_state"]["position"],
            qdot=ob_robot["joint_state"]["velocity"],
            x_goal_0=ob_robot['FullSensor']['goals'][5]['position'],
            weight_goal_0=ob_robot['FullSensor']['goals'][5]['weight'],
            x_goal_1=ob_robot['FullSensor']['goals'][6]['position'],
            weight_goal_1=ob_robot['FullSensor']['goals'][6]['weight'],
            x_obst_0=ob_robot['FullSensor']['obstacles'][2]['position'],
            radius_obst_0=ob_robot['FullSensor']['obstacles'][2]['size'],
            x_obst_1=ob_robot['FullSensor']['obstacles'][3]['position'],
            radius_obst_1=ob_robot['FullSensor']['obstacles'][3]['size'],
            constraint_0=np.array([0, 0, 1, 0.0]),
            x_obst_2=ob_robot['FullSensor']['obstacles'][4]['position'],
            sizes_obst_2=ob_robot['FullSensor']['obstacles'][4]['size'],
            #**arguments,
        )
        action = planner.compute_action(**all_arguments)
        ob, reward, terminated, truncated, info = env.step(action)
        q = ob['robot_0']['joint_state']['position']
        dq = ob['robot_0']['joint_state']['velocity']
        """
        if terminated or truncated:
            print(info)
        """
        vel_mag = np.linalg.norm(dq)
        #print(q[0])
        """
        if terminated or truncated:
            print(info)
            break
        """
    env.close()
    return {}


if __name__ == "__main__":
    render = bool(int(sys.argv[1]))
    res = run_panda_example(render=render, n_steps=1000)
