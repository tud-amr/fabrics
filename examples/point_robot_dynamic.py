import gym
from mpscenes.obstacles.dynamic_sphere_obstacle import DynamicSphereObstacle
import numpy as np
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.goals.goal_composition import GoalComposition
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

# Fabrics example for a 3D point mass robot. The fabrics planner uses a 2D point
# mass to compute actions for a simulated 3D point mass.
#
# todo: tune behavior.

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
        GenericUrdfReacher(urdf="pointRobot.urdf", mode="acc"),
    ]
    env: UrdfEnv  = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    # Set the initial position and velocity of the point mass.
    pos0 = np.array([-2.0, 0.5, 0.0])
    vel0 = np.array([0.1, 0.0, 0.0])
    full_sensor = FullSensor(
            goal_mask=["position", "weight"],
            obstacle_mask=["position", "velocity", "acceleration", "size"],
            variance=0.0,
    )
    # Definition of the obstacle.
    dynamic_obst_dict = {
            "type": "sphere",
            "geometry": {"trajectory": ["-1.0 * t + 5", "0.0 * t", "0.0 * t"], "radius": 0.5},
    }
    obst1 = DynamicSphereObstacle(name="dynamicObst1", content_dict=dynamic_obst_dict)
    dynamic_obst_dict = {
            "type": "sphere",
            "geometry": {"trajectory": ["-0.0 * t + 2", "0.8 * t - 2.0", "0.0 * t"], "radius": 0.7},
    }
    obst2 = DynamicSphereObstacle(name="dynamicObst2", content_dict=dynamic_obst_dict)
    # Definition of the goal.
    goal_dict = {
            "subgoal0": {
                "weight": 0.5,
                "is_primary_goal": True,
                "indices": [0, 1],
                "parent_link" : 0,
                "child_link" : 1,
                "desired_position": [3.5, 0.5],
                "epsilon" : 0.1,
                "type": "staticSubGoal"
            }
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    env.reset(pos=pos0, vel=vel0)
    env.add_sensor(full_sensor, [0])
    env.add_goal(goal.sub_goals()[0])
    env.add_obstacle(obst1)
    env.add_obstacle(obst2)
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
    degrees_of_freedom = 2
    robot_type = "pointRobot"
    # Optional reconfiguration of the planner with collision_geometry/finsler, remove for defaults.
    collision_geometry = "-2.0 / (x ** 1) * xdot ** 2"
    collision_finsler = "1.0/(x**2) * (1 - ca.heaviside(xdot))* xdot**2"
    planner = ParameterizedFabricPlanner(
            degrees_of_freedom,
            robot_type,
            collision_geometry=collision_geometry,
            collision_finsler=collision_finsler
    )
    collision_links = [1]
    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        collision_links=collision_links,
        goal=goal,
        number_obstacles=0,
        number_dynamic_obstacles=2,
        dynamic_obstacle_dimension=2,
    )
    planner.concretize(mode='acc', time_step=0.01)
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
    planner = set_planner(goal)

    action = np.array([0.0, 0.0, 0.0])
    ob, *_ = env.step(action)

    for _ in range(n_steps):
        # Calculate action with the fabric planner, slice the states to drop Z-axis [3] information.
        ob_robot = ob['robot_0']
        arguments = dict(
            q=ob_robot["joint_state"]["position"][0:2],
            qdot=ob_robot["joint_state"]["velocity"][0:2],
            x_goal_0=ob_robot['FullSensor']['goals'][2]['position'][0:2],
            weight_goal_0=ob_robot['FullSensor']['goals'][2]['weight'],
            x_obst_dynamic_0=ob_robot['FullSensor']['obstacles'][3]['position'][0:2],
            xdot_obst_dynamic_0=ob_robot['FullSensor']['obstacles'][3]['velocity'][0:2],
            xddot_obst_dynamic_0=ob_robot['FullSensor']['obstacles'][3]['acceleration'][0:2],
            radius_obst_dynamic_0=ob_robot['FullSensor']['obstacles'][3]['size'],
            x_obst_dynamic_1=ob_robot['FullSensor']['obstacles'][4]['position'][0:2],
            xdot_obst_dynamic_1=ob_robot['FullSensor']['obstacles'][4]['velocity'][0:2],
            xddot_obst_dynamic_1=ob_robot['FullSensor']['obstacles'][4]['acceleration'][0:2],
            radius_obst_dynamic_1=ob_robot['FullSensor']['obstacles'][4]['size'],
            radius_body_1=np.array([0.4])
        )
        action[0:2] = planner.compute_action(**arguments)

        ob, *_, = env.step(action)
    env.close()
    return {}

if __name__ == "__main__":
    res = run_point_robot_urdf(n_steps=10000, render=True)
