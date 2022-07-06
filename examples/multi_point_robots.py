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
        "multi-point-robots-acc-v0",
        dt=0.001,
        render=render,
        number_agents=4,
    )
    q0 = np.array([-1.5, 0, 1.5, 0, 0, 1.5, 0, -1.5])
    q0 += np.random.random(8) * 0.3
    qdot0 = np.array([0.0, 0.0, 0.0, 0.0])
    qdot0 = np.zeros(8)
    initial_observation = env.reset(pos=q0, vel=qdot0)
    # Definition of the obstacle.
    static_obst_dict = {
        "dim": 2,
        "type": "sphere",
        "geometry": {"position": [1.0, 1000.0], "radius": 0.2},
    }
    obst1 = SphereObstacle(name="staticObst", contentDict=static_obst_dict)
    # Definition of the goal.
    weight = 10
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
    obstacles = [obst1]
    env.add_goal(goal0)
    env.add_goal(goal1)
    env.add_goal(goal2)
    env.add_goal(goal3)
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
        "-1.5 / (x ** 2) * xdot ** 2"
    )
    collision_finsler: str = (
        "1.0/(x**2) * (-0.5 * (ca.sign(xdot) - 1)) * xdot**2"
        # "1.0 * xdot**2"
    )
    base_energy: str = "0.5 * 0.01 * ca.dot(xdot, xdot)"
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        robot_type,
        collision_geometry=collision_geometry,
        collision_finsler=collision_finsler,
        damper=damper,
        base_energy=base_energy,
    )
    # The planner hides all the logic behind the function set_components.
    collision_links = [1]
    self_collision_links = {}
    planner.set_components(
        collision_links,
        self_collision_links,
        goal,
        number_obstacles=1,
        number_dynamic_obstacles=3,
        limits=[[-3, 3], [-3, 3]]
    )
    planner.concretize()
    return planner


def compute_action(
    planner, robot_index, ob, goals, obst1
):
    other_positions = []
    other_velocities = []
    other_accelerations = [np.zeros(2),]*3
    for i in range(4):
        if i == robot_index:
            ego_position = ob['x'][2*i:2*i+2]
            ego_velocity = ob['xdot'][2*i:2*i+2]
        else:
            other_positions.append(ob['x'][2*i:2*i+2])
            other_velocities.append(ob['xdot'][2*i:2*i+2])
            #other_velocities.append(np.zeros(2))
    goal = goals[robot_index]
    goal_position = np.array(goal.subGoals()[0].position())
    goal_weight = np.array(goal.subGoals()[0].weight())

    action = planner.compute_action(
        q=ego_position,
        qdot=ego_velocity,
        x_goal_0=goal_position,
        weight_goal_0=goal_weight,
        x_ref_dynamic_obst_0_1_leaf=other_positions[0],
        xdot_ref_dynamic_obst_0_1_leaf=other_velocities[0],
        xddot_ref_dynamic_obst_0_1_leaf=other_accelerations[0],
        x_ref_dynamic_obst_1_1_leaf=other_positions[1],
        xdot_ref_dynamic_obst_1_1_leaf=other_velocities[1],
        xddot_ref_dynamic_obst_1_1_leaf=other_accelerations[1],
        x_ref_dynamic_obst_2_1_leaf=other_positions[2],
        xdot_ref_dynamic_obst_2_1_leaf=other_velocities[2],
        xddot_ref_dynamic_obst_2_1_leaf=other_accelerations[2],
        radius_dynamic_obst_0=np.array([0.15]),
        radius_dynamic_obst_1=np.array([0.15]),
        radius_dynamic_obst_2=np.array([0.15]),
        radius_body_1=np.array([0.20]),
        x_obst_0=np.array(obst1.position()),
        radius_obst_0=np.array([obst1.radius()]),
    )
    action_max = 150
    #action = np.clip(action, np.ones(2) * -action_max, action_max * np.ones(2))
    return action

def compute_all_actions(planner, ob, goals, obst1):
    actions = np.zeros(8)
    for i in range(4):
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
