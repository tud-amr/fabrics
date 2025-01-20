import os
import gymnasium as gym
import numpy as np

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor

from mpscenes.obstacles.box_obstacle import BoxObstacle
from mpscenes.goals.goal_composition import GoalComposition
from fabrics.components.energies.execution_energies import ExecutionLagrangian

from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

# Fabrics example for a 3D point mass robot. The fabrics planner uses a 2D point
# mass to compute actions for a simulated 3D point mass.
#
# todo: tune behavior.
TF = np.identity(4)
TF[0:3,0:3] = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
CAP_LENGTH=1.0
RADIUS = 0.5


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
        GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel"),
    ]
    env: UrdfEnv  = UrdfEnv(
        dt=0.01, robots=robots, render=render
    ).unwrapped
    # Set the initial position and velocity of the point mass.
    pos0 = np.array([-2.0, 3.0, 0.2])
    vel0 = np.array([0.1, 0.0, 0.0])
    full_sensor = FullSensor(
            goal_mask=["position", "weight"],
            obstacle_mask=["position", "size"],
            variance=0.0,
    )
    # Definition of the obstacle.
    static_obst_dict = {
            "type": "box",
            "geometry": {
                "position": [0.0, 2.0, 0.05],
                 "width": 8.0,
                 "length": 1.0,
                 "height": 1.0,
            },
    }
    obst1 = BoxObstacle(name="staticObst1", content_dict=static_obst_dict)
    # Definition of the goal.
    goal_dict = {
            "subgoal0": {
                "weight": 0.5,
                "is_primary_goal": True,
                "indices": [0, 1],
                "parent_link" : 'world',
                "child_link" : 'base_link',
                "desired_position": [3.5, -2.5],
                "epsilon" : 0.1,
                "type": "staticSubGoal"
            }
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    env.reset(pos=pos0, vel=vel0)
    env.add_sensor(full_sensor, [0])
    env.add_goal(goal.sub_goals()[0])
    env.add_obstacle(obst1)
    env.set_spaces()
    return (env, goal)


def set_planner(goal: GoalComposition) -> ParameterizedFabricPlanner:
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
    degrees_of_freedom = 3
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    with open(absolute_path + "/point_robot.urdf", "r", encoding="utf-8") as file:
        urdf = file.read()
    forward_kinematics = GenericURDFFk(
        urdf,
        root_link="world",
        end_links="base_link",
    )
    collision_geometry = "-20.0 / (x ** 1) * xdot ** 2"
    collision_finsler = "1.0/(x**2) * (1 - ca.heaviside(xdot))* xdot**2"
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        forward_kinematics,
        collision_geometry=collision_geometry,
        collision_finsler=collision_finsler
    )
    # The planner hides all the logic behind the function set_components.
    q = planner.variables.position_variable()
    tf_capsule_origin =  forward_kinematics.casadi(
        q, 'base_link', link_transformation=TF
    ) 
    planner.add_capsule_cuboid_geometry(
        "obst_cuboid_0", "capsule_0", tf_capsule_origin, CAP_LENGTH
    )
    planner.set_goal_component(goal)
    execution_energy = ExecutionLagrangian(planner.variables)
    planner.set_execution_energy(execution_energy)
    planner.set_speed_control()

    planner.concretize(mode='vel', time_step=0.01)
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
    planner: ParameterizedFabricPlanner = set_planner(goal)
    link_id = 2
    env.add_collision_link(
            robot_index=0,
            link_index=link_id,
            shape_type="capsule",
            size=[RADIUS, CAP_LENGTH],
            link_transformation=TF,
    )

    action = np.array([0.0, 0.0, 0.0])
    ob, *_ = env.step(action)

    for _ in range(n_steps):
        # Calculate action with the fabric planner, slice the states to drop Z-axis [3] information.
        ob_robot = ob['robot_0']
        arguments = dict(
            q=ob_robot["joint_state"]["position"],
            qdot=ob_robot["joint_state"]["velocity"],
            x_goal_0=ob_robot['FullSensor']['goals'][2]['position'][0:2],
            weight_goal_0=ob_robot['FullSensor']['goals'][2]['weight'],
            x_obst_cuboid_0=ob_robot['FullSensor']['obstacles'][3]['position'],
            size_obst_cuboid_0=ob_robot['FullSensor']['obstacles'][3]['size'],
            radius_capsule_0=RADIUS,
        )
        action = planner.compute_action(**arguments)
        ob, *_, = env.step(action)
    env.close()
    return {}

if __name__ == "__main__":
    res = run_point_robot_urdf(n_steps=10000, render=True)
