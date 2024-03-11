import os
import shutil
import gymnasium as gym
import numpy as np

from robotmodels.utils.robotmodel import RobotModel, LocalRobotModel

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

from urdfenvs.sensors.full_sensor import FullSensor

from urdfenvs.generic_mujoco.generic_mujoco_robot import GenericMujocoRobot
from urdfenvs.generic_mujoco.generic_mujoco_env import GenericMujocoEnv
from urdfenvs.sensors.free_space_decomposition import FreeSpaceDecompositionSensor
from urdfenvs.sensors.free_space_occupancy import FreeSpaceOccupancySensor
from urdfenvs.sensors.full_sensor import FullSensor
from urdfenvs.sensors.lidar import Lidar
from urdfenvs.sensors.sdf_sensor import SDFSensor
from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle

from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner
from urdfenvs.scene_examples.goal import goal1
ROBOTTYPE = 'kinova'
ROBOTMODEL = 'gen3lite'

def initalize_environment(render=True, nr_obst: int = 0):
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
    if os.path.exists(ROBOTTYPE):
        shutil.rmtree(ROBOTTYPE)
    robot_model_original = RobotModel(ROBOTTYPE, ROBOTMODEL)
    robot_model_original.copy_model(os.path.join(os.getcwd(), ROBOTTYPE))
    robot_model = LocalRobotModel(ROBOTTYPE, ROBOTMODEL)

    xml_file = robot_model.get_xml_path()
    robots  = [
        GenericMujocoRobot(xml_file=xml_file, mode="vel"),
    ]
    # Definition of the obstacle.
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [0.5, -0.3, 0.3], "radius": 0.1},
    }
    obst1 = SphereObstacle(name="obstacle_0", content_dict=static_obst_dict)
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [-0.7, 0.0, 0.5], "radius": 0.1},
    }
    obst2 = SphereObstacle(name="obstacle_1", content_dict=static_obst_dict)
    # Definition of the goal.
    goal_dict = {
        "subgoal0": {
            "weight": 1.0,
            "is_primary_goal": True,
            "indices": [0, 1, 2],
            "parent_link": "BASE",
            "child_link": "END_EFFECTOR",
            "desired_position": [-0.24355761, -0.65252747, 0.5],
            "epsilon": 0.05,
            "type": "staticSubGoal",
        },
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    obstacles = [obst1, obst2][0:nr_obst]
    env: GenericMujocoEnv = gym.make(
        'generic-mujoco-env-v0',
        robots=robots,
        obstacles=obstacles,
        goals=goal.sub_goals(),
        sensors=[full_sensor],
        render=render,
        enforce_real_time=True,
    ).unwrapped
    ob, info = env.reset()
    return (env, goal)


def set_planner(goal: GoalComposition, nr_obst: int = 0, degrees_of_freedom: int = 6):
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
    robot_model = RobotModel(ROBOTTYPE, model_name="gen3lite")
    urdf_file = robot_model.get_urdf_path()
    with open(urdf_file, "r", encoding="utf-8") as file:
        urdf = file.read()
    forward_kinematics = GenericURDFFk(
        urdf,
        rootLink="BASE",
        end_link="DUMMY",
    )
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        forward_kinematics,
    )
    collision_links = [
        "FOREARM",
        "DUMMY",
        "END_EFFECTOR"
    ]
    gen3lite_limits = list(np.array([
        [-154.1, 154.1],
        [150.1, 150.1],
        [150.1, 150.1],
        [-148.98, 148.98],
        [-144.97, 145.0],
        [-148.98, 148.98]
    ]) * np.pi/180)
    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        collision_links=collision_links,
        goal=goal,
        number_obstacles=nr_obst,
        number_plane_constraints=1,
        limits=gen3lite_limits,
    )
    planner.concretize(mode='vel', time_step=0.05)
    return planner


def run_kinova_example(n_steps=5000, render=True, dof=6):
    nr_obst=2
    (env, goal) = initalize_environment(render, nr_obst=nr_obst)
    planner = set_planner(goal, nr_obst=nr_obst, degrees_of_freedom=dof)
    # planner.export_as_c("planner.c")
    action = np.zeros(dof)
    ob, *_ = env.step(action)

    for _ in range(n_steps):
        ob_robot = ob['robot_0']
        arguments_dict = dict(
            q=ob_robot["joint_state"]["position"],
            qdot=ob_robot["joint_state"]["velocity"],
            x_goal_0=ob_robot['FullSensor']['goals'][0]['position'],
            weight_goal_0=ob_robot['FullSensor']['goals'][0]['weight'],
            x_obst_0=ob_robot['FullSensor']['obstacles'][0]['position'],
            radius_obst_0=ob_robot['FullSensor']['obstacles'][0]['size'],
            x_obst_1=ob_robot['FullSensor']['obstacles'][1]['position'],
            radius_obst_1=ob_robot['FullSensor']['obstacles'][1]['size'],
            radius_body_END_EFFECTOR = 0.1,
            radius_body_FOREARM = 0.1,
            radius_body_DUMMY = 0.1,
            constraint_0=np.array([0, 0, 1, 0.0]))

        action = planner.compute_action(**arguments_dict)
        ob, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(info)
            break
    env.close()
    return {}

if __name__ == "__main__":
    res = run_kinova_example(n_steps=5000, dof=6)
