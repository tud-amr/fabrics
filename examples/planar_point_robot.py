import gym
import planarenvs.point_robot  # pylint: disable=unused-import

from MotionPlanningGoal.goalComposition import GoalComposition
from MotionPlanningEnv.sphereObstacle import SphereObstacle

import numpy as np
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner


def initalize_environment(render=True):
    """
    Initializes the simulation environment.

    Adds obstacles and goal visualizaion to the environment based and
    steps the simulation once.
    """
    env = gym.make(
        "point-robot-acc-v0", dt=0.05, render=render
    )
    q0 = np.array([4.3, 1.0])
    qdot0 = np.array([-1.0, 0.0])
    initial_observation = env.reset(pos=q0, vel=qdot0)
    # Definition of the obstacle.
    static_obst_dict = {
        "dim": 2,
        "type": "sphere",
        "geometry": {"position": [0.0, 3.5], "radius": 0.6},
    }
    obst1 = SphereObstacle(name="staticObst", contentDict=static_obst_dict)
    static_obst_dict = {
        "dim": 2,
        "type": "sphere",
        "geometry": {"position": [0.0, -1.0], "radius": 0.4},
    }
    obst2 = SphereObstacle(name="staticObst", contentDict=static_obst_dict)
    # Definition of the goal.
    goal_dict = {
        "subgoal0": {
            "m": 2,
            "w": 1.0,
            "prime": True,
            "indices": [0, 1],
            "parent_link": 0,
            "child_link": 2,
            "desired_position": [3.0, 2.0],
            "epsilon": 0.15,
            "type": "staticSubGoal",
        }
    }
    goal = GoalComposition(name="goal", contentDict=goal_dict)
    obstacles = (obst1, obst2)
    env.add_goal(goal)
    env.add_obstacle(obst1)
    env.add_obstacle(obst2)
    return (env, obstacles, goal, initial_observation)


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
        number_obstacles=2,
    )
    planner.concretize()
    return planner


def run_point_robot_example(n_steps=5000, render=True):
    (env, obstacles, goal, initial_observation) = initalize_environment(
        render=render
    )
    ob = initial_observation
    obst1 = obstacles[0]
    obst2 = obstacles[1]
    planner = set_planner(goal)

    # Start the simulation
    print("Starting simulation")
    x = 1
    sub_goal_0_position = np.array(goal.subGoals()[0].position())
    #sub_goal_0_position = np.array(goal.subGoals()[0].position())
    sub_goal_0_weight = np.array(goal.subGoals()[0].weight())
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
        )
        ob, *_ = env.step(action)
    return {}


if __name__ == "__main__":
    res = run_point_robot_example(n_steps=5000)
