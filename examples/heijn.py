import os
import gymnasium as gym
import numpy as np

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor

from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle

from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

absolute_path = os.path.dirname(os.path.abspath(__file__))
urdf_file = absolute_path + "/heijn_robot.urdf" 

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
        GenericUrdfReacher(urdf=urdf_file, mode="acc"),
    ]
    env: UrdfEnv = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    ).unwrapped.unwrapped
    # Set the initial position and velocity of the robot.
    pos0 = np.array([-2.0, 0.5, 0.9])
    vel0 = np.array([0.1, 0.0, 0.0])
    full_sensor = FullSensor(
            goal_mask=["position", "weight"],
            obstacle_mask=['position', 'size'],
            variance=0.0
    )
    # Definition of the obstacle.
    static_obst_dict_1 = {
            "type": "sphere",
            "geometry": {"position": [2.0, 0.2, 0.0], "radius": 0.6},
    }
    static_obst_dict_2 = {
            "type": "sphere",
            "geometry": {"position": [1.3, 2.3, 0.0], "radius": 0.8},
    }
    static_obst_dict_3 = {
            "type": "sphere",
            "geometry": {"position": [1.0, -2.1, 0.0], "radius": 0.3},
    }
    obst1 = SphereObstacle(name="staticObst1", content_dict=static_obst_dict_1)
    obst2 = SphereObstacle(name="staticObst2", content_dict=static_obst_dict_2)
    obst3 = SphereObstacle(name="staticObst3", content_dict=static_obst_dict_3)
    obstacles = (obst1, obst2, obst3) # Add additional obstacles here.
    # Definition of the goal.
    goal_dict = {
            "subgoal0": {
                "weight": 2.5,
                "is_primary_goal": True,
                "indices": [0, 1],
                "parent_link" : 'odom',
                "child_link" : 'front_link',
                "desired_position": [3.5, 0.5],
                "epsilon" : 0.1,
                "type": "staticSubGoal"
            }
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    env.reset(pos=pos0, vel=vel0)
    env.add_sensor(full_sensor, [0])
    for obst in obstacles:
        env.add_obstacle(obst)
    for sub_goal in goal.sub_goals():
        env.add_goal(sub_goal)
    env.set_spaces()
    return (env, goal)


def set_planner(goal: GoalComposition):
    """
    Initializes the fabric planner for the heijn robot.

    This function defines the forward kinematics for collision avoidance,
    and goal reaching. These components are fed into the fabrics planner.

    In the top section of this function, an example for optional reconfiguration
    can be found. Commented by default.

    Params
    ----------
    goal: StaticSubGoal
        The goal to the motion planning problem.
    """
    degrees_of_freedom = 3
    # Optional reconfiguration of the planner with collision_geometry/finsler, remove for defaults.
    collision_geometry = "-1.0 / (x ** 1) * xdot ** 2"
    collision_finsler = "1.0/(x**1) * (1 - ca.heaviside(xdot))* xdot**2"
    with open(urdf_file, 'r') as file:
        urdf = file.read()
    forward_kinematics = GenericURDFFk(
        urdf,
        root_link='odom',
        end_links=[
            'front_link',
            'collision_link_front_right',
            'collision_link_front_left',
            'collision_link_center_right',
            'collision_link_center_left',
            'collision_link_rear_right',
            'collision_link_rear_left',
        ],
    )
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        forward_kinematics,
        collision_geometry=collision_geometry,
        collision_finsler=collision_finsler
    )
    collision_links = [
        'collision_link_front_right',
        'collision_link_front_left',
        'collision_link_center_right',
        'collision_link_center_left',
        'collision_link_rear_right',
        'collision_link_rear_left',
    ]
    self_collision_links = {}
    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        collision_links=collision_links,
        goal=goal,
        number_obstacles=3,
    )
    planner.concretize()
    return planner


def run_heijn_robot(n_steps=10000, render=True):
    """
    Set the gym environment, the planner and run heijn robot example.
    
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

    for _ in range(n_steps):
        # Calculate action with the fabric planner, slice the states to drop Z-axis [3] information.
        ob_robot = ob['robot_0']
        action = planner.compute_action(
            q=ob_robot['joint_state']['position'],
            qdot=ob_robot['joint_state']['velocity'],
            x_goal_0=ob_robot['FullSensor']['goals'][5]['position'][0:2],
            weight_goal_0=ob_robot['FullSensor']['goals'][5]['weight'],
            radius_body_collision_link_front_right=np.array([0.12]),
            radius_body_collision_link_front_left=np.array([0.12]),
            radius_body_collision_link_center_right=np.array([0.18]),
            radius_body_collision_link_center_left=np.array([0.18]),
            radius_body_collision_link_rear_right=np.array([0.12]),
            radius_body_collision_link_rear_left=np.array([0.12]),
            x_obst_0=ob_robot['FullSensor']['obstacles'][2]['position'],
            radius_obst_0=ob_robot['FullSensor']['obstacles'][2]['size'],
            x_obst_1=ob_robot['FullSensor']['obstacles'][3]['position'],
            radius_obst_1=ob_robot['FullSensor']['obstacles'][3]['size'],
            x_obst_2=ob_robot['FullSensor']['obstacles'][4]['position'],
            radius_obst_2=ob_robot['FullSensor']['obstacles'][4]['size'],
        )
        ob, *_, = env.step(action)
    env.close()
    return {}


if __name__ == "__main__":
    res = run_heijn_robot(n_steps=10000, render=True)



