import os
import shutil
import gymnasium as gym
import numpy as np

from robotmodels.utils.robotmodel import RobotModel, LocalRobotModel

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

from urdfenvs.sensors.full_sensor import FullSensor

from urdfenvs.generic_mujoco.generic_mujoco_robot import GenericMujocoRobot
from urdfenvs.generic_mujoco.generic_mujoco_env import GenericMujocoEnv

from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle

from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

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
            "parent_link": "panda_link0",
            "child_link": "panda_hand",
            "desired_position": [0.1, -0.6, 0.4],
            "epsilon": 0.05,
            "type": "staticSubGoal",
        },
        "subgoal1": {
            "weight": 5.0,
            "is_primary_goal": False,
            "indices": [0, 1, 2],
            "parent_link": "panda_link7",
            "child_link": "panda_hand",
            "desired_position": [0.1, 0.0, 0.0],
            "epsilon": 0.05,
            "type": "staticSubGoal",
        }
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    obstacles = [obst1, obst2]
    env = GenericMujocoEnv(
        robots=robots,
        obstacles=obstacles,
        goals=goal.sub_goals(),
        sensors=[full_sensor],
        render=render,
        enforce_real_time=True,
    )
    env.reset()
    return (env, goal)


def set_planner(goal: GoalComposition, degrees_of_freedom: int = 7):
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
    #     forward_kinematics,
    #     base_inertia=base_inertia,
    #     attractor_potential=attractor_potential,
    #     damper=damper,
    # )
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
    collision_links = ['panda_link7', 'panda_link3', 'panda_link4']
    self_collision_pairs = {"panda_link7": ['panda_link3', 'panda_link4', 'panda_link2', 'panda_link1']}
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
        self_collision_pairs=self_collision_pairs,
        goal=goal,
        number_obstacles=2,
        number_plane_constraints=1,
        limits=panda_limits,
    )
    planner.concretize(mode='vel', time_step=0.05)
    return planner


def run_panda_example(n_steps=5000, render=True):
    (env, goal) = initalize_environment(render)
    planner = set_planner(goal)
    # planner.export_as_c("planner.c")
    action = np.zeros(7)
    ob, *_ = env.step(action)
    body_links={1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 7: 0.1}
    """
    for body_link, radius in body_links.items():
        env.add_collision_link(0, body_link, shape_type='sphere', size=[radius])
    """


    for _ in range(n_steps):
        ob_robot = ob['robot_0']
        action = planner.compute_action(
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
            radius_body_links=body_links,
            constraint_0=np.array([0, 0, 1, 0.0]),
        )
        ob, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(info)
            break
    env.close()
    return {}


if __name__ == "__main__":
    res = run_panda_example(n_steps=5000)
