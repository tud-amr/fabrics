import os
from typing import Tuple
import gymnasium as gym
import numpy as np

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor

from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from fabrics.components.energies.execution_energies import ExecutionLagrangian

from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

absolute_path = os.path.dirname(os.path.abspath(__file__))
URDF_FILE = absolute_path + "/panda_collision_links.urdf"
CAPSULE_LINKS = list(range(2,4)) + list(range(5,9))



def setup_collision_links_panda(i) -> Tuple[np.ndarray, str, int, float, float]:
    link_translations = [
            np.array([0.0, 0.0, -0.1915]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, -0.145]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, -0.26]),
            np.array([0, 0.08, -0.13]),
            np.array([0, 0.0, -0.03]),
            np.array([0, 0.0, 0.01]),
            np.array([0.0424, 0.0424, -0.0250]),
    ]
    link_rotations = [np.identity(3)] * 9
    link_rotations[8] = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    links = [0, 1, 2, 3, 4, 4, 5, 6, 7]
    link_names = [f"panda_link{link_id+1}" for link_id in links]
    lengths = [0.2830, 0.12, 0.15, 0.12, 0.1, 0.14, 0.08, 0.14, 0.01]
    radii = [0.09, 0.09, 0.09, 0.09, 0.09, 0.055, 0.08, 0.07, 0.06]
    tf = np.identity(4)
    tf[0:3,0:3] = link_rotations[i]
    tf[0:3, 3] = link_translations[i]
    return (tf, link_names[i], links[i], radii[i], lengths[i])

def initalize_environment(render=True):
    """
    Initializes the simulation environment.

    Adds obstacles and goal visualizaion to the environment based and
    steps the simulation once.
    """
    robots = [
        GenericUrdfReacher(urdf=URDF_FILE, mode="acc"),
    ]
    env: UrdfEnv  = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    ).unwrapped
    full_sensor = FullSensor(
            goal_mask=["position", "weight"],
            obstacle_mask=["position", "size"],
            variance=0.0
    )
    # Definition of the obstacle.
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [0.40, -0.25, 0.5], "radius": 0.1},
    }
    obst1 = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
    # Definition of the goal.
    goal_dict = {
        "subgoal0": {
            "weight": 1.0,
            "is_primary_goal": True,
            "indices": [0, 1, 2],
            "parent_link": "panda_link0",
            "child_link": "panda_link8",
            "desired_position": [0.1, -0.6, 0.4],
            "epsilon": 0.05,
            "type": "staticSubGoal",
        },
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    obstacles = [obst1]
    env.reset()
    env.add_sensor(full_sensor, [0])
    for obst in obstacles:
        env.add_obstacle(obst)
    for sub_goal in goal.sub_goals():
        env.add_goal(sub_goal)
    env.set_spaces()
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
    with open(URDF_FILE, "r", encoding="utf-8") as file:
        urdf = file.read()
    forward_kinematics = GenericURDFFk(
        urdf,
        rootLink="panda_link0",
        end_link="panda_link8",
    )
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        forward_kinematics,
    )
    q = planner.variables.position_variable()

    for i in CAPSULE_LINKS:
        tf, link_name, _, _, length = setup_collision_links_panda(i)
        tf_capsule_origin =  forward_kinematics.casadi(
            q, "panda_link0", link_name, tf
        ) 
        planner.add_capsule_sphere_geometry(
            "obst_1", f"capsule_{i}", tf_capsule_origin, length
        )


    planner.set_goal_component(goal)
    execution_energy = ExecutionLagrangian(planner.variables)
    planner.set_execution_energy(execution_energy)
    planner.set_speed_control()

    planner.concretize()
    return planner


def run_panda_capsule_example(n_steps=5000, render=True):
    (env, goal) = initalize_environment(render)
    planner = set_planner(goal)
    action = np.zeros(7)
    ob, *_ = env.step(action)
    static_args = {}
    for i in CAPSULE_LINKS:
        tf, _, link_id, radius, length = setup_collision_links_panda(i)
        env.add_collision_link(
                robot_index=0,
                link_index=link_id,
                shape_type="capsule",
                size=[radius, length],
                link_transformation=tf,
        )
        static_args[f"radius_capsule_{i}"] = radius

    for _ in range(n_steps):
        ob_robot = ob["robot_0"]
        args = dict(
            q=ob_robot["joint_state"]["position"],
            qdot=ob_robot["joint_state"]["velocity"],
            x_goal_0=ob_robot["FullSensor"]["goals"][3]["position"],
            weight_goal_0=ob_robot["FullSensor"]["goals"][3]["weight"],
            x_obst_1=ob_robot["FullSensor"]["obstacles"][2]["position"],
            radius_obst_1=ob_robot["FullSensor"]["obstacles"][2]["size"],
            **static_args
        )
        action = planner.compute_action(**args)
        ob, *_ = env.step(action)
    env.close()
    return {}


if __name__ == "__main__":
    res = run_panda_capsule_example(n_steps=5000)
