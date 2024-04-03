import os
import gymnasium as gym
import numpy as np
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor

from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from robotmodels.utils.robotmodel import RobotModel, LocalRobotModel
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

"""
Experiments with Kuka iiwa
If 
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [0.3, -0.3, 0.5], "radius": 0.1},
    }
for the first obstacle, fabrics ends up in a local minima and is unable to
recover from it. This would be a good case to improve with imitation learning.
"""

ROBOTTYPE = 'kinova'
ROBOTMODEL = 'gen3_6dof'

def initalize_environment(render=True, nr_obst: int = 0):
    """
    Initializes the simulation environment.

    Adds obstacles and goal visualizaion to the environment based and
    steps the simulation once.
    """
    robot_model = RobotModel('kinova', model_name='gen3_6dof')
    urdf_file = robot_model.get_urdf_path()
    robots = [
        GenericUrdfReacher(urdf=urdf_file, mode="acc"),
    ]
    env: UrdfEnv = UrdfEnv(
        robots=robots,
        dt=0.01,
        render=render,
        observation_checking=False,
    )
    full_sensor = FullSensor(
            goal_mask=["position", "weight"],
            obstacle_mask=['position', 'size'],
            variance=0.0
    )
    # Definition of the obstacle.
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [0.3, -0.3, 0.3], "radius": 0.1},
    }
    obst1 = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [-0.7, 0.0, 0.5], "radius": 0.1},
    }
    obst2 = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
    goal_dict = {
        "subgoal0": {
            "weight": 1.0,
            "is_primary_goal": True,
            "indices": [0, 1, 2],
            "parent_link": "base_link",
            "child_link": "end_effector_link",
            "desired_position": [-0.24355761, -0.75252747, 0.5],
            "epsilon": 0.05,
            "type": "staticSubGoal",
        },
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    obstacles = [obst1, obst2][0:nr_obst]
    pos0 = np.array([0.0, 0.8, -1.5, 2.0, 0.0, 0.0])
    env.reset(pos=pos0)
    env.add_sensor(full_sensor, [0])
    for obst in obstacles:
        env.add_obstacle(obst)
    for sub_goal in goal.sub_goals():
        env.add_goal(sub_goal)
    env.set_spaces()
    collision_radii = {1:0.1, 2:0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1}
    for collision_link_nr in collision_radii.keys():
         env.add_collision_link(0, collision_link_nr, shape_type='sphere', size=[0.10])
    return (env, goal)


def set_planner(goal: GoalComposition, nr_obst: int = 0, degrees_of_freedom: int = 6):
    """
    Initializes the fabric planner for the kuka robot.

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
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    robot_model = RobotModel('kinova', model_name='gen3_6dof')
    urdf_file = robot_model.get_urdf_path()
    with open(urdf_file, "r", encoding="utf-8") as file:
        urdf = file.read()
    forward_kinematics = GenericURDFFk(
        urdf,
        rootLink="base_link",
        end_link="end_effector_link",
    )
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        forward_kinematics,
    )
    collision_links = [
        "forearm_link",
        "spherical_wrist_1_link",
        "spherical_wrist_2_link",
        "bracelet_link",
        "end_effector_link"
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
    planner.concretize()
    return planner


def run_kinova_example(n_steps=5000, render=True, dof=6):
    nr_obst = 2
    (env, goal) = initalize_environment(render, nr_obst=nr_obst)
    planner = set_planner(goal, nr_obst, degrees_of_freedom=dof)
    action = np.zeros(dof)
    ob, *_ = env.step(action)

    rot_matrix = np.array([[-0.339, -0.784306, -0.51956],
                           [-0.0851341, 0.57557, -0.813309],
                           [0.936926, -0.23148, -0.261889]])

    for w in range(n_steps):
        ob_robot = ob['robot_0']

        arguments_dict = dict(
            q=ob_robot["joint_state"]["position"],
            qdot=ob_robot["joint_state"]["velocity"],
            x_goal_0=ob_robot['FullSensor']['goals'][nr_obst+2]['position'],
            weight_goal_0=ob_robot['FullSensor']['goals'][nr_obst+2]['weight'],
            x_obst_0=ob_robot['FullSensor']['obstacles'][nr_obst]['position'],
            radius_obst_0=ob_robot['FullSensor']['obstacles'][nr_obst]['size'],
            x_obst_1=ob_robot['FullSensor']['obstacles'][nr_obst+1]['position'],
            radius_obst_1=ob_robot['FullSensor']['obstacles'][nr_obst+1]['size'],
            radius_body_end_effector_link = 0.1,
            radius_body_bracelet_link = 0.1,
            radius_body_spherical_wrist_1_link = 0.1,
            radius_body_spherical_wrist_2_link = 0.1,
            radius_body_forearm_link=0.1,
            constraint_0=np.array([0, 0, 1, 0.0]))

        action = planner.compute_action(**arguments_dict)
        ob, *_ = env.step(action)
    env.close()
    return {}


if __name__ == "__main__":
    dof = 6
    res = run_kinova_example(n_steps=5000, dof=dof)
