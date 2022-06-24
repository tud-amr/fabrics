import gym
import planarenvs.multi_point_robots

from MotionPlanningGoal.goalComposition import GoalComposition
from MotionPlanningEnv.sphereObstacle import SphereObstacle
from MotionPlanningEnv.dynamicSphereObstacle import DynamicSphereObstacle

import numpy as np
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner


def initalize_environment(render=True):
    """
    Initializes the simulation environment.

    Adds obstacles and goal visualizaion to the environment based and
    steps the simulation once.
    """
    env = gym.make(
        "multi-point-robots-acc-v0", dt=0.01, render=render, number_agents=2,
    )
    q0 = np.array([3.0, 1.00, 0.0, 1.0])
    qdot0 = np.array([0.0, 0.0, 0.0, 0.0])
    initial_observation = env.reset(pos=q0, vel=qdot0)
    # Definition of the obstacle.
    static_obst_dict = {
        "dim": 2,
        "type": "analyticSphere",
        "geometry": {"trajectory": ["0.0 * t", "3.5"], "radius": 0.6},
    }
    obst1 = DynamicSphereObstacle(name="staticObst", contentDict=static_obst_dict)
    # Definition of the goal.
    goal_dict = {
        "subgoal0": {
            "m": 2,
            "w": 1.0,
            "prime": True,
            "indices": [0, 1],
            "parent_link": 0,
            "child_link": 2,
            "desired_position": [0.0, 1.0],
            "epsilon": 0.15,
            "type": "staticSubGoal",
        }
    }
    goal1 = GoalComposition(name="goal", contentDict=goal_dict)
    goal_dict = {
        "subgoal0": {
            "m": 2,
            "w": 1.0,
            "prime": True,
            "indices": [0, 1],
            "parent_link": 0,
            "child_link": 2,
            "desired_position": [3.0, 1.0],
            "epsilon": 0.15,
            "type": "staticSubGoal",
        }
    }
    goal2 = GoalComposition(name="goal", contentDict=goal_dict)
    goals = [goal1, goal2]
    obstacles = [obst1]
    env.add_goal(goal1)
    env.add_goal(goal2)
    env.add_obstacle(obst1)
    return (env, obstacles, goals, initial_observation)


def set_planner(goal: GoalComposition):
    """
    Initializes the fabric planner for the point robot.

    This function defines the forward kinematics for collision avoidance,
    and goal reaching. These components are fed into the fabrics planner.

    In the top section of this function, an example for optional reconfiguration
    can be found. Commented by default.

    """
    degrees_of_freedom = 2
    robot_type = 'pointRobot'

    ## Optional reconfiguration of the planner
    # base_inertia = 0.03
    # attractor_potential = "20 * ca.norm_2(x)**4"
    # damper = {
    #     "alpha_b": 0.5,
    #     "alpha_eta": 0.5,
    #     "alpha_shift": 0.5,
    #     "beta_distant": 0.01,
    #     "beta_close": 6.5,
    #     "radius_shift": 0.1,
    # }
    # planner = ParameterizedFabricPlanner(
    #     degrees_of_freedom,
    #     robot_type,
    #     base_inertia=base_inertia,
    #     attractor_potential=attractor_potential,
    #     damper=damper,
    # )
    planner = ParameterizedFabricPlanner(degrees_of_freedom, robot_type)
    # The planner hides all the logic behind the function set_components.
    collision_links = [1]
    self_collision_links = {}
    planner.set_components(
        collision_links,
        self_collision_links,
        goal,
        number_obstacles=1,
    )
    planner.concretize()
    return planner

def compute_action(planner, ob, sub_goal_0_position, sub_goal_0_weight, obst1_position, obst1_radius):
    action = planner.compute_action(
        q=ob["x"],
        qdot=ob["xdot"],
        x_goal_0=sub_goal_0_position,
        weight_goal_0=sub_goal_0_weight,
        x_obst_0=obst1_position,
        radius_obst_0=np.array([obst1_radius]),
        radius_body_1=np.array([0.02]),
    )
    return action


def run_point_robot_example(n_steps=5000, render=True):
    (env, obstacles, goals, initial_observation) = initalize_environment(
        render=render
    )
    ob = initial_observation
    obst1 = obstacles[0]
    planner = set_planner(goals[0])

    # Start the simulation
    print("Starting simulation")
    sub_goal_0_position = [0, ] * 2
    sub_goal_0_weight = [0, ] * 2
    sub_goal_0_position[0] = np.array(goals[0].subGoals()[0].position())
    sub_goal_0_weight[0] = np.array(goals[0].subGoals()[0].weight())
    sub_goal_0_position[1] = np.array(goals[1].subGoals()[0].position())
    sub_goal_0_weight[1] = np.array(goals[1].subGoals()[0].weight())
    obst1_position = np.array(obst1.position())
    ob_0 = {}
    ob_1 = {}
    for _ in range(n_steps):
        ob_0['x'] = ob['x'][0:2]
        ob_1['x'] = ob['x'][2:4]
        ob_0['xdot'] = ob['xdot'][0:2]
        ob_1['xdot'] = ob['xdot'][2:4] 
        obst1_position = ob_1['x']
        action_0 = compute_action(planner, ob_0, sub_goal_0_position[0], sub_goal_0_weight[0], obst1_position, obst1.radius())
        obst1_position = ob_0['x']
        action_1 = compute_action(planner, ob_1, sub_goal_0_position[1], sub_goal_0_weight[1], obst1_position, obst1.radius())
        action = np.concatenate((action_0, action_1))
        ob, *_ = env.step(action)
    return {}


if __name__ == "__main__":
    res = run_point_robot_example(n_steps=5000)
