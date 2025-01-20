import os
import gymnasium as gym
import numpy as np
import time

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

from robotmodels.utils.robotmodel import RobotModel
from urdfenvs.generic_mujoco.generic_mujoco_env import GenericMujocoEnv
from urdfenvs.generic_mujoco.generic_mujoco_robot import GenericMujocoRobot
from urdfenvs.sensors.full_sensor import FullSensor

from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.goals.goal_composition import GoalComposition

from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

# Fabrics example for a 3D point mass robot. The fabrics planner uses a 2D point
# mass to compute actions for a simulated 3D point mass.
#
# todo: tune behavior.

DT = 0.01

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
    '''
    robots = [
        GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel"),
    ]
    env: UrdfEnv  = UrdfEnv(
        dt=0.01, robots=robots, render=render
    ).unwrapped
    '''
    robot_model = RobotModel('pointRobot', 'pointRobot')
    xml_file = robot_model.get_xml_path()
    robots  = [
        GenericMujocoRobot(xml_file=xml_file, mode="vel"),
    ]
    full_sensor = FullSensor(
            goal_mask=["position", "weight"],
            obstacle_mask=["position", "size"],
            variance=0.0,
            physics_engine_name='mujoco',
    )
    static_obst_dict = {
            "type": "sphere",
            "geometry": {"position": [2.0, 0.0, 0.0], "radius": 1.0},
    }
    obst1 = SphereObstacle(name="staticObst1", content_dict=static_obst_dict)
    goal_dict = {
            "subgoal0": {
                "weight": 2.5,
                "is_primary_goal": True,
                "indices": [0, 1],
                "parent_link" : 'world',
                "child_link" : 'base_link',
                "desired_position": [3.5, 0.5],
                "epsilon" : 0.1,
                "type": "staticSubGoal"
            }
    }
    augmented_goal_dict = {
            "subgoal0": {
                "weight": 2.5,
                "is_primary_goal": True,
                "indices": [0, 1, 2],
                "parent_link" : 'world',
                "child_link" : 'base_link',
                "desired_position": [3.5, 0.5, 0.0],
                "epsilon" : 0.1,
                "type": "staticSubGoal"
            }
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    augmented_goal = GoalComposition(name='goal_aug', content_dict=augmented_goal_dict)
    goal_list = [augmented_goal.sub_goals()[0]]
    obstacle_list = [obst1]
    sensors = [full_sensor]
    env: GenericMujocoEnv = GenericMujocoEnv(
        robots=robots,
        obstacles=obstacle_list,
        goals=goal_list,
        sensors=sensors,
        render=render,
        enforce_real_time=True,
    ).unwrapped

    # Set the initial position and velocity of the point mass.
    pos0 = np.array([-2.0, 0.5, 0.0])
    vel0 = np.array([0.1, 0.0, 0.0])
    # Definition of the obstacle.
    # Definition of the goal.
    env.reset(pos=pos0, vel=vel0)
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
    degrees_of_freedom = 3
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    with open(absolute_path + "/point_robot.urdf", "r", encoding="utf-8") as file:
        urdf = file.read()
    forward_kinematics = GenericURDFFk(
        urdf,
        root_link="world",
        end_links="base_link",
    )
    # collision_geometry = "-2.0 / (x ** 1) * xdot ** 2"
    collision_geometry = "-2*ca.log(1 + ca.exp(-100*x+50)) * xdot ** 2"
    # collision_finsler = "1.0/(x**2) * (1 - ca.heaviside(xdot))* xdot**2"
    # collision_finsler = "2*ca.log(1 + ca.exp(-10*x+50)) * xdot ** 2"
    collision_finsler = "2*ca.log(1 + ca.exp(-100*x+50)) * xdot ** 2"
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        forward_kinematics,
        collision_geometry=collision_geometry,
        collision_finsler=collision_finsler
    )
    collision_links = ["base_link"]
    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        collision_links=collision_links,
        goal=goal,
        number_obstacles=1,
    )
    planner.concretize(mode='vel', time_step=DT)
    return planner


def run_point_robot_urdf(n_steps=10000, render=False, enforce_real_time=True):
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
        t0 = time.perf_counter()
        # Calculate action with the fabric planner, slice the states to drop Z-axis [3] information.
        ob_robot = ob['robot_0']
        pos_obs = ob_robot['FullSensor']['obstacles'][0]['position'] #[-1.5, 0.5, 0.] #
        action = planner.compute_action(
            q=ob_robot["joint_state"]["position"],
            qdot=ob_robot["joint_state"]["velocity"],
            x_goal_0=ob_robot['FullSensor']['goals'][0]['position'][0:2],
            weight_goal_0=ob_robot['FullSensor']['goals'][0]['weight'],
            x_obst_0=pos_obs,
            radius_obst_0=1., #ob_robot['FullSensor']['obstacles'][0]['size'],
            radius_body_base_link=np.array([0.2])
        )
        t1 = time.perf_counter()
        ob, *_, = env.step(action)
        print("action: ", action)
        t2 = time.perf_counter()
        #print(f"Time step was actually {t2-t0}")
        #print(f"Time for compute was {t1-t0}")
        distance = np.linalg.norm(ob_robot["joint_state"]["position"][0:2] - pos_obs[0:2]) - 1.0 - 0.2
        print("distance=", distance)
        if distance < 0.:
            print("collision occured!!")
        # print(f"Time for render was {t2-t1}")
    env.close()
    return {}

if __name__ == "__main__":
    res = run_point_robot_urdf(n_steps=2000, render=True)
