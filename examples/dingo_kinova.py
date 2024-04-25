import os
import gymnasium as gym
import numpy as np
import quaternionic
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor

from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.obstacles.box_obstacle import BoxObstacle
from robotmodels.utils.robotmodel import RobotModel, LocalRobotModel
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner


robot_model = RobotModel('dingo_kinova', model_name='dingo_kinova')
URDF_FILE = robot_model.get_urdf_path()


def initalize_environment(render=True, nr_obst: int = 0, n_cube_obst:int = 8):
    """
    Initializes the simulation environment.

    Adds obstacles and goal visualizaion to the environment based and
    steps the simulation once.
    """
    # robot_model = RobotModel('kinova', model_name='gen3_6dof')
    urdf_file = URDF_FILE
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
    radius_ring = 0.3
    obstacles = []
    obstacle_resolution = n_cube_obst -1
    goal_orientation = [-0.366, 0.0, 0.0, 0.3305]
    rotation_matrix = quaternionic.array(goal_orientation).to_rotation_matrix
    whole_position = [0.1, 0.6, 0.8]
    for i in range(obstacle_resolution):
        angle = i/obstacle_resolution * 2.*np.pi
        origin_position = [
            0.0,
            radius_ring * np.cos(angle),
            radius_ring * np.sin(angle),
        ]
        position = np.dot(np.transpose(rotation_matrix), origin_position) + whole_position
        static_obst_dict = {
            "type": "box",
            "geometry": {"position": position.tolist(), "length": 0.1, "width": 0.1, "height": 0.1},
        }
        obstacles.append(BoxObstacle(name="staticObst", content_dict=static_obst_dict))

    static_obst_dict = {
            "type": "box",
            "geometry": {"position": [0.0, 0.0, 0.1], "length": 0.4, "width": 0.4, "height": 0.2},
        }
    obstacles.append(BoxObstacle(name="staticObst", content_dict=static_obst_dict))

    goal_dict = {
        "subgoal0": {
            "weight": 1.0,
            "is_primary_goal": True,
            "indices": [0, 1, 2],
            "parent_link": "world",
            "child_link": "arm_tool_frame",
            "desired_position": whole_position,
            "epsilon": 0.05,
            "type": "staticSubGoal",
        },
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    
    pos0 = np.array([-1.0, -1.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    env.reset(pos=pos0)
    env.add_sensor(full_sensor, [0])
    for obst in obstacles:
        env.add_obstacle(obst)
    for sub_goal in goal.sub_goals():
        env.add_goal(sub_goal)
    env.set_spaces()
    collision_radii = {3:0.35, 12: 0.1, 13:0.1, 14:0.1, 15: 0.1, 16: 0.1, 17: 0.1}
    for collision_link_nr in collision_radii.keys():
         env.add_collision_link(0, collision_link_nr, shape_type='sphere', size=[collision_radii[collision_link_nr]])
    return (env, goal)


def set_planner(goal: GoalComposition, n_obst: int, n_cube_obst:int, degrees_of_freedom: int):
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
    # robot_model = RobotModel('kinova', model_name='gen3_6dof')
    urdf_file = URDF_FILE
    with open(urdf_file, "r", encoding="utf-8") as file:
        urdf = file.read()
    forward_kinematics = GenericURDFFk(
        urdf,
        rootLink="world",
        end_link="arm_tool_frame",
    )
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        forward_kinematics,
    )
    collision_links = [
        "base_link_y",
        "arm_arm_link",
        "arm_forearm_link",
        "arm_lower_wrist_link",
        "arm_upper_wrist_link",
        "arm_end_effector_link"
    ]
    # 3 omnibase joints + 6 arm joints
    omnibase_limits = np.array([
        [-4, 4],
        [-4, 4],
        [-np.pi, np.pi]])
    kinova_gen3lite_limits = np.array([
        [-154.1, 154.1],
        [150.1, 150.1],
        [150.1, 150.1],
        [-148.98, 148.98],
        [-144.97, 145.0],
        [-148.98, 148.98]]) * np.pi/180
    joint_limits = list(np.concatenate((omnibase_limits, kinova_gen3lite_limits)))

    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        collision_links=collision_links,
        goal=goal,
        number_obstacles=n_obst,
        number_obstacles_cuboid=n_cube_obst,
        number_plane_constraints=0,
        limits=joint_limits,
    )
    planner.concretize()
    return planner


def run_kinova_example(n_steps=5000, render=True, dof=3+6):
    nr_obst = 0
    n_cube_obst = 8
    total_obst = nr_obst + n_cube_obst
    (env, goal) = initalize_environment(render, nr_obst=nr_obst, n_cube_obst=n_cube_obst)
    planner = set_planner(goal, n_obst=nr_obst, n_cube_obst=n_cube_obst, degrees_of_freedom=dof)
    action = np.zeros(dof)
    ob, *_ = env.step(action)

    for w in range(n_steps):
        ob_robot = ob['robot_0']
        #  x_obsts_cuboid is a np.array of shape (n_cube_obst, 3)
        #  size_obsts_cuboid is a np.array of shape (n_cube_obst, 3), 3: length", "width", "height ?
        x_obsts = [ob_robot['FullSensor']['obstacles'][i+2]['position'] for i in range(n_cube_obst)]
        size_obsts = [ob_robot['FullSensor']['obstacles'][i+2]['size'] for i in range(n_cube_obst)]

        arguments_dict = dict(
            q=ob_robot["joint_state"]["position"],
            qdot=ob_robot["joint_state"]["velocity"],
            x_obsts_cuboid=x_obsts,
            size_obsts_cuboid=size_obsts,
            x_goal_0=ob_robot['FullSensor']['goals'][total_obst+2]['position'],
            weight_goal_0=ob_robot['FullSensor']['goals'][total_obst+2]['weight'],
            radius_body_base_link_y = 0.35,
            radius_body_arm_arm_link = 0.1,
            radius_body_arm_forearm_link = 0.1,
            radius_body_arm_lower_wrist_link = 0.1,
            radius_body_arm_upper_wrist_link = 0.1,
            radius_body_arm_end_effector_link=0.1,
            # constraint_0=np.array([0, 0, 1, 0.0])
            )

        action = planner.compute_action(**arguments_dict)
        ob, *_ = env.step(action)
    env.close()
    return {}


if __name__ == "__main__":
    dof = 9
    res = run_kinova_example(n_steps=5000, dof=dof)
