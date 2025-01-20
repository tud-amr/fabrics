import os
import time
import gymnasium as gym
import numpy as np

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor
from urdfenvs.sensors.free_space_decomposition import FreeSpaceDecompositionSensor

from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.obstacles.box_obstacle import BoxObstacle
from mpscenes.goals.goal_composition import GoalComposition

from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

NUMBER_OF_RAYS = 10
NUMBER_OF_CONSTRAINTS = 10


def get_goal_fsd():
    goal_dict = {
            "subgoal0": {
                "weight": 5,
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

def get_obstacles_fsd():
    static_obst_dict = {
            "type": "sphere",
            "geometry": {"position": [2.0, -0.5, 0.15], "radius": 0.8},
    }
    obst1 = SphereObstacle(name="staticObst1", content_dict=static_obst_dict)
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [0.0, 1.0, 0.15], "radius": 0.3},
    }
    obst2 = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [3.0, 1.2, 0.15], "radius": 0.3},
    }
    obst3 = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
    static_obst_dict = {
        "type": "box",
        "geometry": {"position": [2.0, 1.7, 0.15], "width": 0.3, "length": 0.2, "height": 0.3},
    }
    obst4 = BoxObstacle(name="staticObst", content_dict=static_obst_dict)
    return [obst1, obst2, obst3, obst4]

def initalize_environment(render):
    """
    Initializes the simulation environment.

    Adds an obstacle and goal visualizaion to the environment and
    steps the simulation once.
   j
    Params
    ----------
    render
        Boolean toggle to set rendering on (True) or off (False).
    """
    robots = [
        GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel"),
    ]
    env: UrdfEnv  = UrdfEnv(
        dt=0.01, robots=robots, render=render
    ).unwrapped
    # Set the initial position and velocity of the point mass.
    pos0 = np.array([-2.0, 0.5, 0.0])
    vel0 = np.array([0.1, 0.0, 0.0])
    full_sensor = FullSensor(
        goal_mask=["position", "weight"],
        obstacle_mask=["position", "size"],
        variance=0.0,
    )
    fsd_sensor = FreeSpaceDecompositionSensor(
            'lidar_sensor_joint',
            max_radius=5,
            plotting_interval=100,
            nb_rays=NUMBER_OF_RAYS,
            number_constraints=NUMBER_OF_CONSTRAINTS,
    )
    # Definition of the obstacle.
    obstacles = get_obstacles_fsd()
    # Definition of the goal.
    goal = get_goal_fsd()
    env.reset(pos=pos0, vel=vel0)
    env.add_sensor(full_sensor, [0])
    env.add_sensor(fsd_sensor, [0])
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
    goal: StaticSubGoal
        The goal to the motion planning problem.
    """
    collision_geometry = "-2.0 / (x ** 1) * xdot ** 2"
    collision_finsler = "1.0/(x**2) * (1 - ca.heaviside(xdot))* xdot**2"
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    with open(absolute_path + "/point_robot.urdf", "r", encoding="utf-8") as file:
        urdf = file.read()
    collision_links = ["base_link"]
    degrees_of_freedom = 3
    forward_kinematics = GenericURDFFk(
        urdf,
        root_link="world",
        end_links="base_link",
    )
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        forward_kinematics,
        geometry_plane_constraint=collision_geometry,
        finsler_plane_constraint=collision_finsler
    )
    collision_links = ["base_link"]
    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        collision_links=collision_links,
        goal=goal,
        number_obstacles=0,
        number_plane_constraints=NUMBER_OF_CONSTRAINTS,
    )
    planner.concretize(mode="vel", time_step=0.01)
    return planner


def run_point_robot_urdf(n_steps=10000, render=True):
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
    env.reconfigure_camera(5, 0, 270.1, [0, 0, 0])
    planner = set_planner(goal)

    action = np.array([0.0, 0.0, 0.0])
    ob, *_ = env.step(action)

    for _ in range(n_steps):
        t0 = time.perf_counter()
        # Calculate action with the fabric planner, slice the states to drop Z-axis [3] information.
        ob_robot = ob['robot_0']

        arguments = dict(
            q=ob_robot["joint_state"]["position"],
            qdot=ob_robot["joint_state"]["velocity"],
            x_goal_0=ob_robot['FullSensor']['goals'][2]['position'][0:2],
            weight_goal_0=ob_robot['FullSensor']['goals'][2]['weight'],
            radius_body_base_link=np.array([0.35]),
        )
        for i in range(NUMBER_OF_CONSTRAINTS):
            arguments[f"constraint_{i}"] = ob_robot["FreeSpaceDecompSensor"][f"constraint_{i}"]
        action = planner.compute_action(**arguments)
        ob, *_, = env.step(action)
        t1 = time.perf_counter()
        print(t1-t0)
    return {}

if __name__ == "__main__":
    res = run_point_robot_urdf(n_steps=10000, render=True)
