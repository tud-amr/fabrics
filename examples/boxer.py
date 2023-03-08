import gym
import logging
import numpy as np
from urdfenvs.robots.boxer import BoxerRobot
from urdfenvs.sensors.full_sensor import FullSensor
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.goals.goal_composition import GoalComposition
from fabrics.planner.non_holonomic_parameterized_planner import NonHolonomicParameterizedFabricPlanner

logging.basicConfig(level=logging.INFO)
"""
Fabrics example for the boxer robot.
"""

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
        BoxerRobot(mode="acc"),
    ]
    env: UrdfEnv  = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    full_sensor = FullSensor(goal_mask=["position"], obstacle_mask=["position", "radius"])
    # Definition of the obstacle.
    static_obst_dict = {
            "type": "sphere",
            "geometry": {"position": [1.0, 0.0, 0.0], "radius": 1.0},
    }
    obst1 = SphereObstacle(name="staticObst1", content_dict=static_obst_dict)
    obstacles = [obst1] # Add additional obstacles here.
    # Definition of the goal.
    goal_dict = {
            "subgoal0": {
                "weight": 1.0,
                "is_primary_goal": True,
                "indices": [0, 1],
                "parent_link" : 'origin',
                "child_link" : 'ee_link',
                "desired_position": [4.0, -0.2],
                "epsilon" : 0.1,
                "type": "staticSubGoal"
            }
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)

    pos0 = np.array([-4.0, 0.4, 0.0])
    vel0 = np.array([0.0, 0.0])
    env.reset(pos=pos0, vel=vel0)
    env.add_sensor(full_sensor, [0])
    for obst in obstacles:
        env.add_obstacle(obst)
    for sub_goal in goal.sub_goals():
        env.add_goal(sub_goal)
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
    robot_type = "boxer"
    # Optional reconfiguration of the planner with collision_geometry/finsler, remove for defaults.
    collision_geometry = "-2.0 / (x ** 1) * xdot ** 2"
    collision_finsler = "1.0/(x**1) * (1 - ca.heaviside(xdot))* xdot**2"
    planner = NonHolonomicParameterizedFabricPlanner(
            degrees_of_freedom,
            robot_type,
            collision_geometry=collision_geometry,
            collision_finsler=collision_finsler,
            l_offset="0.1/ca.norm_2(xdot)",
    )
    collision_links = ["ee_link"]
    self_collision_links = {}
    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        collision_links,
        self_collision_links,
        goal,
        number_obstacles=1,
    )
    planner.concretize()
    return planner


def run_boxer_example(n_steps=10000, render=True):
    """
    Set the gym environment, the planner and run point robot example.
    
    Params
    ----------
    n_steps
        Total number of simulation steps.
    render
        Boolean toggle to set rendering on (True) or off (False).
    """
    (env, goal) = initalize_environment(render)
    planner = set_planner(goal)
    action = np.zeros(7)
    ob, *_ = env.step(action)
    env.reconfigure_camera(3.000001907348633, 8.800007820129395, -64.20002746582031, (0.0, 0.0, 0.0))

    for _ in range(n_steps):
        ob_robot = ob['robot_0']
        qudot = np.array([
            ob_robot['joint_state']['forward_velocity'][0],
            ob_robot['joint_state']['velocity'][2]
        ])
        action = planner.compute_action(
            q=ob_robot["joint_state"]["position"],
            qdot=ob_robot["joint_state"]["velocity"],
            qudot=qudot,
            x_goal_0=ob_robot['FullSensor']['goals'][0][0][0:2],
            weight_goal_0=goal.sub_goals()[0].weight(),
            m_rot=0.2,
            m_base_x=1.5,
            m_base_y=1.5,
            x_obst_0=ob_robot['FullSensor']['obstacles'][0][0],
            radius_obst_0=ob_robot['FullSensor']['obstacles'][0][1],
            radius_body_ee_link=0.5,
        )

        ob, *_, = env.step(action)
    return {}


if __name__ == "__main__":
    res = run_boxer_example(n_steps=10000, render=True)



