import pdb
import gym
import numpy as np
import urdfenvs.albert_reacher
import logging
from urdfenvs.albert_reacher.envs.acc import AlbertReacherAccEnv
from MotionPlanningEnv.sphereObstacle import SphereObstacle
from MotionPlanningGoal.goalComposition import GoalComposition
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
    env: AlbertReacherAccEnv = gym.make("albert-reacher-acc-v0", dt=0.05, render=render)
    initial_observation = env.reset()
    # Definition of the obstacle.
    static_obst_dict = {
            "type": "sphere",
            "geometry": {"position": [3.0, 0.0, 0.0], "radius": 1.0},
    }
    obst1 = SphereObstacle(name="staticObst1", content_dict=static_obst_dict)
    obstacles = (obst1) # Add additional obstacles here.
    # Definition of the goal.
    goal_dict = {
        "subgoal0": {
            "weight": 1.0,
            "is_primary_goal": True,
            "indices": [0, 1, 2],
            "parent_link" : 'origin',
            "child_link" : 'panda_hand',
            "desired_position": [4.0, -0.2, 1.0],
            "epsilon" : 0.1,
            "type": "staticSubGoal"
        },
        "subgoal1": {
            "weight": 5.0,
            "is_primary_goal": False,
            "indices": [0, 1, 2],
            "parent_link": "panda_link7",
            "child_link": "panda_hand",
            "desired_position": [0.0, 0.0, -0.107],
            "epsilon": 0.05,
            "type": "staticSubGoal",
        }
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    # Add walls, the goal and the obstacle to the environment.
    #env.add_walls([0.1, 10, 0.5], [[5.0, 0, 0], [-5.0, 0.0, 0.0], [0.0, 5.0, np.pi/2], [0.0, -5.0, np.pi/2]])
    env.add_goal(goal)
    env.add_obstacle(obst1)
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
    degrees_of_freedom = 10
    robot_type = "albert"
    # Optional reconfiguration of the planner with collision_geometry/finsler, remove for defaults.
    collision_geometry = "-2.0 / (x ** 1) * xdot ** 2"
    collision_finsler = "1.0/(x**1) * (1 - ca.heaviside(xdot))* xdot**2"
    planner = NonHolonomicParameterizedFabricPlanner(
            degrees_of_freedom,
            robot_type,
            collision_geometry=collision_geometry,
            collision_finsler=collision_finsler,
            l_offset="0.5",
    )
    collision_links = ["base_link", "base_tip_link", 'panda_link1', 'panda_link4', 'panda_link6', 'panda_hand']
    self_collision_pairs = {}
    boxer_limits = [
            [-10, 10],
            [-10, 10],
            [-6 * np.pi, 6 * np.pi],
            [-2.8973, 2.8973],
            [-1.7628, 1.7628],
            [-2.8973, 2.8973],
            [-3.0718, -0.0698],
            [-2.8973, 2.8973],
            [-0.0175, 3.7525],
            [-2.8973, 2.8973]
        ]
    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        collision_links,
        self_collision_pairs,
        goal,
        limits=boxer_limits,
        number_obstacles=1,
    )
    planner.concretize()
    return planner


def run_albert_reacher_example(n_steps=10000, render=True):
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
    obst1 = obstacles
    print(f"Initial observation : {ob}")
    planner = set_planner(goal)
    # Start the simulation.
    print("Starting simulation")
    sub_goal_0_position = np.array(goal.sub_goals()[0].position())
    sub_goal_0_weight = goal.sub_goals()[0].weight()
    sub_goal_1_position = np.array(goal.sub_goals()[1].position())
    sub_goal_1_weight = goal.sub_goals()[1].weight()
    obst1_position = np.array(obst1.position())
    for _ in range(n_steps):
        # Calculate action with the fabric planner, slice the states to drop Z-axis [3] information.
        qudot = np.array([ob['joint_state']['forward_velocity'][0], ob['joint_state']['velocity'][2]])
        qudot = np.concatenate((qudot, ob['joint_state']['velocity'][3:]))
        action = planner.compute_action(
            q=ob["joint_state"]["position"],
            qdot=ob["joint_state"]["velocity"],
            qudot=qudot,
            x_goal_0=sub_goal_0_position,
            weight_goal_0=sub_goal_0_weight,
            x_goal_1=sub_goal_1_position,
            weight_goal_1=sub_goal_1_weight,
            m_rot=1.0,
            m_base_x=2.5,
            m_base_y=2.5,
            m_arm=10.0,
            x_obst_0=obst1_position,
            radius_obst_0=np.array([obst1.radius()]),
            radius_body_base_link=0.8,
            radius_body_base_tip_link=0.3,
            radius_body_panda_link1=0.1,
            radius_body_panda_link4=0.1,
            radius_body_panda_link6=0.15,
            radius_body_panda_hand=0.1,
        )
        ob, *_, = env.step(action)
    return {}


if __name__ == "__main__":
    res = run_albert_reacher_example(n_steps=10000, render=True)



