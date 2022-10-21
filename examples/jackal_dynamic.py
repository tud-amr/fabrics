import gym
import numpy as np
from urdfenvs.robots.jackal import JackalRobot
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
import logging
from MotionPlanningEnv.sphereObstacle import SphereObstacle
from MotionPlanningEnv.dynamicSphereObstacle import DynamicSphereObstacle
from MotionPlanningGoal.goalComposition import GoalComposition
from fabrics.planner.non_holonomic_parameterized_planner import NonHolonomicParameterizedFabricPlanner

logging.basicConfig(level=logging.INFO)
"""
Fabrics example for the jackal robot.
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
        JackalRobot(mode="acc"),
    ]
    env: UrdfEnv  = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    # Set the initial position and velocity of the point mass.
    pos0 = np.array([-2.0, 0.4, 0.0])
    vel0 = np.array([0.0, 0.0])
    initial_observation = env.reset(pos=pos0, vel=vel0)
    # Definition of the obstacle.
    static_obst_dict = {
            "type": "sphere",
            "geometry": {"position": [1.5, 0.0, 0.0], "radius": 1.0},
    }
    static_obst = SphereObstacle(name="staticObst1", content_dict=static_obst_dict)
    dynamic_obst_dict = {
        "type": "analyticSphere",
        "geometry": {"trajectory": ["3 - 0.0 * t", "2.0 * ca.sin(0.5 * t)"], "radius": 0.4},
    }
    dynamic_obst = DynamicSphereObstacle(name="staticObst", content_dict=dynamic_obst_dict)
    obstacles = (static_obst, dynamic_obst) # Add additional obstacles here.
    # Definition of the goal.
    goal_dict = {
            "subgoal0": {
                "weight": 1.0,
                "is_primary_goal": True,
                "indices": [0, 1],
                "parent_link" : 'origin',
                "child_link" : 'ee_link',
                "desired_position": [3.2, -0.0],
                "epsilon" : 0.1,
                "type": "staticSubGoal"
            }
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    # Add walls, the goal and the obstacle to the environment.
    #env.add_walls([0.1, 10, 0.5], [[5.0, 0, 0], [-5.0, 0.0, 0.0], [0.0, 5.0, np.pi/2], [0.0, -5.0, np.pi/2]])
    env.add_goal(goal)
    env.add_obstacle(static_obst)
    env.add_obstacle(dynamic_obst)
    return (env, obstacles, goal, initial_observation)


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
    robot_type = "jackal"
    # Optional reconfiguration of the planner with collision_geometry/finsler, remove for defaults.
    collision_geometry = "-2.0 / (x ** 1) * xdot ** 2"
    collision_finsler = "5.0/(x**1) * (1 - ca.heaviside(xdot))* xdot**2"
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
        number_dynamic_obstacles=1,
    )
    planner.concretize()
    return planner


def run_dynamic_jackal_example(n_steps=10000, render=True):
    """
    Set the gym environment, the planner and run point robot example.
    
    Params
    ----------
    n_steps
        Total number of simulation steps.
    render
        Boolean toggle to set rendering on (True) or off (False).
    """
    (env, obstacles, goal, initial_observation) = initalize_environment(render)
    ob = initial_observation
    static_obstacle, dynamic_obstacle = obstacles
    print(f"Initial observation : {ob}")
    planner = set_planner(goal)
    # Start the simulation.
    print("Starting simulation")
    sub_goal_0_position = np.array(goal.sub_goals()[0].position())
    sub_goal_0_weight = goal.sub_goals()[0].weight()
    input("Wait to start.")
    for _ in range(n_steps):
        # Calculate action with the fabric planner, slice the states to drop Z-axis [3] information.
        qudot = np.array([
            ob["robot_0"]['joint_state']['forward_velocity'][0],
            ob["robot_0"]['joint_state']['velocity'][2]
        ])
        action = planner.compute_action(
            q=ob["robot_0"]["joint_state"]["position"],
            qdot=ob["robot_0"]["joint_state"]["velocity"],
            qudot=qudot,
            x_goal_0=sub_goal_0_position,
            m_rot=0.3,
            m_base_x=1.0,
            m_base_y=1.0,
            weight_goal_0=sub_goal_0_weight,
            x_obst_0=static_obstacle.position()[0:2],
            radius_obst_0=static_obstacle.radius(),
            radius_dynamic_obst_0=dynamic_obstacle.radius(),
            x_ref_dynamic_obst_0_ee_link_leaf=dynamic_obstacle.position(t=env.t()),
            xdot_ref_dynamic_obst_0_ee_link_leaf=dynamic_obstacle.velocity(t=env.t()),
            xddot_ref_dynamic_obst_0_ee_link_leaf=dynamic_obstacle.acceleration(t=env.t()),
            radius_body_ee_link=0.5,
        )

        ob, *_, = env.step(action)
    return {}


if __name__ == "__main__":
    res = run_dynamic_jackal_example(n_steps=10000, render=True)



