import gym
import planarenvs.n_link_reacher  # pylint: disable=unused-import

from MotionPlanningGoal.goalComposition import GoalComposition
from MotionPlanningEnv.sphereObstacle import SphereObstacle
from forwardkinematics.planarFks.planarArmFk import PlanarArmFk

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
    initial_observation = env.reset()
    # Definition of the obstacle.
    static_obst_dict = {
        "dim": 2,
        "type": "sphere",
        "geometry": {"position": [0.9, -0.5], "radius": 0.2},
    }
    obst1 = SphereObstacle(name="staticObst", contentDict=static_obst_dict)
    static_obst_dict = {
        "dim": 2,
        "type": "sphere",
        "geometry": {"position": [-1.0, -1.0], "radius": 0.1},
    }
    obst2 = SphereObstacle(name="staticObst", contentDict=static_obst_dict)
    # Definition of the prime goal.
    goal_dict = {
        "subgoal0": {
            "m": 2,
            "w": 1.0,
            "prime": True,
            "indices": [0, 1],
            "parent_link": 0,
            "child_link": 3,
            "desired_position": [1.5, 1.0],
            "epsilon": 0.15,
            "type": "staticSubGoal",
        },
        "subgoal1": {
            "m": 1,
            "w": 3.0,
            "prime": False,
            "indices": [1],
            "parent_link": 2,
            "child_link": 3,
            "angle": -0.3,
            "desired_position": [0.0],
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


def set_planner(goal: GoalComposition, degrees_of_freedom: int = 3):
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
    q = planner.variables.position_variable()
    planar_arm_fk = PlanarArmFk(degrees_of_freedom)
    forward_kinematics = []
    for i in range(1, degrees_of_freedom + 1):
        forward_kinematics.append(planar_arm_fk.fk(q, i, positionOnly=True))
    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        forward_kinematics,
        goal,
        number_obstacles=2,
    )
    planner.concretize()
    return planner


def run_planar_arm_example(n_steps=5000, render=True):
    degrees_of_freedom = 3
    (env, obstacles, goal, initial_observation) = initalize_environment(
        degrees_of_freedom=degrees_of_freedom, render=render
    )
    ob = initial_observation
    obst1 = obstacles[0]
    obst2 = obstacles[1]
    planner = set_planner(goal, degrees_of_freedom=degrees_of_freedom)

    # Start the simulation
    print("Starting simulation")
    sub_goal_0_position = np.array(goal.subGoals()[0].position())
    sub_goal_1_position = np.array(goal.subGoals()[1].position())
    sub_goal_0_weight = np.array([goal.subGoals()[0].weight()])
    sub_goal_1_weight = np.array([goal.subGoals()[1].weight()])
    obst1_position = np.array(obst1.position())
    obst2_position = np.array(obst2.position())
    for _ in range(n_steps):
        action = planner.compute_action(
            q=ob["x"],
            qdot=ob["xdot"],
            x_goal_0=sub_goal_0_position,
            x_goal_1=sub_goal_1_position,
            weight_goal_0=sub_goal_0_weight,
            weight_goal_1=sub_goal_1_weight,
            x_obst_0=obst2_position,
            x_obst_1=obst1_position,
            radius_obst_0=np.array([obst1.radius()]),
            radius_obst_1=np.array([obst2.radius()]),
            radius_body=np.array([0.02]),
        )
        ob, *_ = env.step(action)
    return {}


if __name__ == "__main__":
    res = run_planar_arm_example(n_steps=5000)
