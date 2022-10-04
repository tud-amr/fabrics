import gym
import sys
import planarenvs.n_link_reacher  # pylint: disable=unused-import

from MotionPlanningGoal.goalComposition import GoalComposition
from MotionPlanningEnv.sphereObstacle import SphereObstacle

import numpy as np
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner


def initalize_environment(degrees_of_freedom=3, render=True):
    """
    Initializes the simulation environment.

    Adds obstacles and goal visualizaion to the environment based and
    steps the simulation once.
    """
    env = gym.make(
        "nLink-reacher-acc-v0", dt=0.05, n=degrees_of_freedom, render=render
    )
    initial_observation = env.reset(pos=np.random.random(degrees_of_freedom) * 0.1)
    # Definition of the obstacle.
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [3.9, -0.5], "radius": 0.2},
    }
    obst1 = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [-1.0, -1.0], "radius": 0.1},
    }
    obst2 = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
    # Definition of the prime goal.
    goal_dict = {
        "subgoal0": {
            "weight": 2.0,
            "is_primary_goal": True,
            "indices": [0, 1],
            "parent_link": 0,
            "child_link": 3,
            "desired_position": [1.5, 1.0],
            "epsilon": 0.15,
            "type": "staticSubGoal",
        },
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    obstacles = (obst1, obst2)
    env.add_goal(goal)
    env.add_obstacle(obst1)
    env.add_obstacle(obst2)
    return (env, obstacles, goal, initial_observation)


def set_planner(goal: GoalComposition, degrees_of_freedom: int = 3, use_limits: bool = False):
    """
    Initializes the fabric planner for the planar arm robot.

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

    robot_type = 'planarArm'

    ## Optional reconfiguration of the planner
    # base_inertia = 0.03
    # attractor_potential = "sym('k_attractor') * ca.norm_2(x)**4"
    # planner = ParameterizedFabricPlanner(
    #     degrees_of_freedom,
    #     robot_type,
    #     base_inertia=base_inertia,
    #     attractor_potential=attractor_potential,
    #     damper=damper,
    # )
    planner = ParameterizedFabricPlanner(degrees_of_freedom, robot_type)
    if use_limits:
        joint_limits = [[-2.0, 2.0], [-0.5, 0.5], [-2.0, 2.0]]
    else:
        joint_limits = []
    collision_links = range(1, degrees_of_freedom + 1)
    self_collision_links = {}
    planner.set_components(
        collision_links,
        self_collision_links,
        goal=goal,
        limits=joint_limits,
        number_obstacles=2,
    )
    planner.concretize()
    return planner


def run_planar_arm_limits_example(n_steps=5000, render=True, use_limits: bool = False):
    print(f"Running example with limits? {use_limits}")
    degrees_of_freedom = 3
    (env, obstacles, goal, initial_observation) = initalize_environment(
        degrees_of_freedom=degrees_of_freedom, render=render
    )
    ob = initial_observation
    obst1 = obstacles[0]
    obst2 = obstacles[1]
    planner = set_planner(goal, degrees_of_freedom=degrees_of_freedom, use_limits=use_limits)

    # Start the simulation
    print("Starting simulation")
    sub_goal_0_position = np.array(goal.sub_goals()[0].position())
    sub_goal_0_weight = np.array([goal.sub_goals()[0].weight()])
    obst1_position = np.array(obst1.position())
    obst2_position = np.array(obst2.position())
    for _ in range(n_steps):
        action = planner.compute_action(
            q=ob["x"],
            qdot=ob["xdot"],
            x_goal_0=sub_goal_0_position,
            weight_goal_0=sub_goal_0_weight,
            x_obst_0=obst2_position,
            x_obst_1=obst1_position,
            radius_obst_0=np.array([obst1.radius()]),
            radius_obst_1=np.array([obst2.radius()]),
            radius_body_1=np.array([0.02]),
            radius_body_2=np.array([0.02]),
            radius_body_3=np.array([0.02]),
        )
        ob, *_ = env.step(action)
    return {}


if __name__ == "__main__":
    arguments = sys.argv
    if len(arguments) == 1 or arguments[0] == 'use_limits':
        use_limits = True
    else:
        use_limits = False
    res = run_planar_arm_limits_example(n_steps=5000, use_limits = use_limits)
