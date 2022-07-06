import gym
import time
import planarenvs.multi_point_robots

from MotionPlanningGoal.goalComposition import GoalComposition
from MotionPlanningEnv.sphereObstacle import SphereObstacle
from MotionPlanningEnv.dynamicSphereObstacle import DynamicSphereObstacle

import numpy as np
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

number_agents = 4
ignorant_agent = 7
initial_position_noise = 0.3


def initalize_environment(render=True):
    """
    Initializes the simulation environment.

    Adds obstacles and goal visualizaion to the environment based and
    steps the simulation once.
    """
    env = gym.make(
        "multi-point-robots-acc-v0",
        dt=0.01,
        render=render,
        number_agents=number_agents,
    )
    if number_agents == 4:
        q0 = np.array([-1.5, 0, 1.5, 0, 0, 1.5, 0, -1.5])
    if number_agents == 3:
        q0 = np.array([-1.5, 0, 1.5, 0, 0, 1.5])
    if number_agents == 2:
        q0 = np.array([-1.5, 0, 1.5, 0])
    q0 += np.random.random(2*number_agents) * initial_position_noise
    qdot0 = np.zeros(2*number_agents)
    initial_observation = env.reset(pos=q0, vel=qdot0)
    # Definition of the obstacle.
    static_obst_dict = {
        "dim": 2,
        "type": "sphere",
        "geometry": {"position": [1.0, 1000.0], "radius": 0.2},
    }
    obst1 = SphereObstacle(name="staticObst", contentDict=static_obst_dict)
    # Definition of the goal.
    weight = 2
    goal_dict = {
        "subgoal0": {
            "m": 2,
            "w": weight,
            "prime": True,
            "indices": [0, 1],
            "parent_link": 0,
            "child_link": 2,
            "desired_position": [1.5, 0.0],
            "epsilon": 0.05,
            "type": "staticSubGoal",
        }
    }
    goal0 = GoalComposition(name="goal", contentDict=goal_dict)
    goal_dict = {
        "subgoal0": {
            "m": 2,
            "w": weight,
            "prime": True,
            "indices": [0, 1],
            "parent_link": 0,
            "child_link": 2,
            "desired_position": [-1.5, 0.0],
            "epsilon": 0.05,
            "type": "staticSubGoal",
        }
    }
    goal1 = GoalComposition(name="goal", contentDict=goal_dict)
    goal_dict = {
        "subgoal0": {
            "m": 2,
            "w": weight,
            "prime": True,
            "indices": [0, 1],
            "parent_link": 0,
            "child_link": 2,
            "desired_position": [0.0, -1.5],
            "epsilon": 0.05,
            "type": "staticSubGoal",
        }
    }
    goal2 = GoalComposition(name="goal", contentDict=goal_dict)
    goal_dict = {
        "subgoal0": {
            "m": 2,
            "w": weight,
            "prime": True,
            "indices": [0, 1],
            "parent_link": 0,
            "child_link": 2,
            "desired_position": [0.0, 1.5],
            "epsilon": 0.05,
            "type": "staticSubGoal",
        }
    }
    goal3 = GoalComposition(name="goal", contentDict=goal_dict)
    goals = [goal0, goal1, goal2, goal3]
    goals = goals[0:number_agents]
    obstacles = [obst1]
    for goal in goals:
        env.add_goal(goal)
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
    robot_type = "pointRobot"

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
    damper = {
        "alpha_b": 0.5,
        "alpha_eta": 0.5,
        "alpha_shift": 0.5,
        "beta_distant": 0.01,
        "beta_close": 10.0,
        "radius_shift": 0.3,
    }
    collision_geometry: str = (
        "-1.5 / (x ** 1) * xdot ** 2"
    )
    collision_finsler: str = (
        "1.0/(x**1) * (-0.5 * (ca.sign(xdot) - 1)) * xdot**2"
        # "1.0 * xdot**2"
    )
    base_energy: str = "0.5 * 0.01 * ca.dot(xdot, xdot)"
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        robot_type,
        collision_geometry=collision_geometry,
        collision_finsler=collision_finsler,
        #damper=damper,
        #base_energy=base_energy,
    )
    # The planner hides all the logic behind the function set_components.
    collision_links = [1]
    self_collision_links = {}
    planner.set_components(
        collision_links,
        self_collision_links,
        goal,
        number_obstacles=1,
        number_dynamic_obstacles=number_agents-1,
        limits=[[-3, 3], [-3, 3]]
    )
    planner.concretize()
    return planner


def compute_action(
    planner, robot_index, ob, goals, obst1
):
    other_positions = []
    other_velocities = []
    other_accelerations = [np.zeros(2),]*(number_agents - 1)
    for i in range(number_agents):
        if i == robot_index:
            ego_position = ob['x'][2*i:2*i+2]
            ego_velocity = ob['xdot'][2*i:2*i+2]
        else:
            if robot_index == ignorant_agent:
                other_positions.append(np.ones(2) * 10000)
                other_velocities.append(np.zeros(2))
            else:
                other_positions.append(ob['x'][2*i:2*i+2])
                other_velocities.append(ob['xdot'][2*i:2*i+2])
            #other_velocities.append(np.zeros(2))
    arguments = {}
    for i in range(number_agents-1):
        arguments[f"x_ref_dynamic_obst_{i}_1_leaf"] = other_positions[i]
        arguments[f"xdot_ref_dynamic_obst_{i}_1_leaf"] = other_velocities[i]
        arguments[f"xddot_ref_dynamic_obst_{i}_1_leaf"] = other_accelerations[i]
        arguments[f"radius_dynamic_obst_{i}"] = np.array([0.05])
    goal = goals[robot_index]
    goal_position = np.array(goal.subGoals()[0].position())
    goal_weight = np.array(goal.subGoals()[0].weight())

    t0 = time.perf_counter()
    action = planner.compute_action(
        q=ego_position,
        qdot=ego_velocity,
        x_goal_0=goal_position,
        weight_goal_0=goal_weight,
        radius_body_1=np.array([0.10]),
        x_obst_0=np.array(obst1.position()),
        radius_obst_0=np.array([obst1.radius()]),
        **arguments
    )
    t1 = time.perf_counter()
    print(f"computation time : {(t1-t0)*1e3}")
    action_max = 150
    #action = np.clip(action, np.ones(2) * -action_max, action_max * np.ones(2))
    return action

def compute_all_actions(planner, ob, goals, obst1):
    actions = np.zeros(2*number_agents)
    for i in range(number_agents):
        actions[2*i:2*i+2] = compute_action(
            planner,
            i,
            ob,
            goals,
            obst1,
        )
    return actions


def run_point_robot_example(n_steps=5000, render=True):
    (env, obstacles, goals, initial_observation) = initalize_environment(render=render)
    ob = initial_observation
    obst1 = obstacles[0]
    planner = set_planner(goals[0])

    # Start the simulation
    print("Starting simulation")
    for _ in range(n_steps):
        action = compute_all_actions(planner, ob, goals, obst1)
        ob, *_ = env.step(action)
    return {}


if __name__ == "__main__":
    res = run_point_robot_example(n_steps=5000)
