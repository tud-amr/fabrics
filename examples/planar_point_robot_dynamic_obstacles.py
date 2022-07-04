import gym
import sys
import planarenvs.point_robot  # pylint: disable=unused-import

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
        "point-robot-acc-v0", dt=0.01, render=render
    )
    q0 = np.array([-2.6, 0.5])
    qdot0 = np.array([-0.0, 0.0])
    initial_observation = env.reset(pos=q0, vel=qdot0)
    # Definition of the obstacle.
    static_obst_dict = {
        "dim": 2,
        "type": "sphere",
        "geometry": {"trajectory": ["-3.0", "6 - 1.5 * t"], "radius": 0.6},
    }
    obst1 = DynamicSphereObstacle(name="staticObst", contentDict=static_obst_dict)
    static_obst_dict = {
        "dim": 2,
        "type": "analyticSphere",
        "geometry": {"trajectory": ["-3 + 0.5 * t + 0.1 * t**2", "0.0"], "radius": 0.4},
    }
    obst2 = DynamicSphereObstacle(name="staticObst", contentDict=static_obst_dict)
    # Definition of the goal.
    goal_dict = {
        "subgoal0": {
            "m": 2,
            "w": 2.5,
            "prime": True,
            "indices": [0, 1],
            "parent_link": 0,
            "child_link": 2,
            "desired_position": [-3.0, 0.0],
            "epsilon": 0.15,
            "type": "staticSubGoal",
        }
    }
    goal = GoalComposition(name="goal", contentDict=goal_dict)
    obstacles = (obst1, obst2)
    env.add_goal(goal)
    env.add_obstacle(obst1)
    #env.add_obstacle(obst2)
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
    collision_geometry: str = (
        "3.5 / (x ** 1) * (-0.5 * (ca.sign(xdot) - 1))  * xdot ** 2"
    )
    collision_finsler: str = (
        "5.0/(x**1) * (-0.5 * (ca.sign(xdot) - 1)) * xdot**2"
    )
    base_energy: str = (
        "0.5 * 0.02 * ca.dot(xdot, xdot)"
    )
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        robot_type,
        collision_geometry=collision_geometry,
        #collision_finsler=collision_finsler,
        #base_energy=base_energy,
    )
    # The planner hides all the logic behind the function set_components.
    collision_links = [1]
    self_collision_links = {}
    joint_limits = [[-4.0, 4.0], [-4.0, 4.0]]
    planner.set_components(
        collision_links,
        self_collision_links,
        None,
        number_obstacles=0,
        number_dynamic_obstacles=1,
        limits=joint_limits
    )
    planner.concretize()
    return planner


def run_point_robot_example(n_steps=5000, render=True, dynamic_fabric=True):
    (env, obstacles, goal, initial_observation) = initalize_environment(
        render=render
    )
    ob = initial_observation
    obst1 = obstacles[0]
    obst2 = obstacles[1]
    planner = set_planner(goal)

    if not dynamic_fabric:
        print(f"Assuming zero velocity for the obstacle")
    else:
        print(f"Respecting the velocity and acceleration of the obstacle")

    # Start the simulation
    print("Starting simulation")
    x = 1
    sub_goal_0_position = np.array(goal.subGoals()[0].position())
    #sub_goal_0_position = np.array(goal.subGoals()[0].position())
    sub_goal_0_weight = np.array(goal.subGoals()[0].weight())
    for _ in range(n_steps):
        obst1_position = np.array(obst1.position(t=env.t()))
        obst1_velocity = np.array(obst1.velocity(t=env.t()))
        obst1_acceleration = np.array(obst1.acceleration(t=env.t()))
        obst2_position = np.array(obst2.position(t=env.t()))
        obst2_velocity = np.array(obst2.velocity(t=env.t()))
        obst2_acceleration = np.array(obst2.acceleration(t=env.t()))
        if not dynamic_fabric:
            obst1_velocity *= 0
            obst1_acceleration *= 0
            obst2_velocity *= 0
            obst2_acceleration *= 0
        action = planner.compute_action(
            q=ob["x"],
            qdot=ob["xdot"],
            x_goal_0=sub_goal_0_position,
            weight_goal_0=sub_goal_0_weight,
            radius_dynamic_obst_0=np.array([obst1.radius()]),
            radius_dynamic_obst_1=np.array([obst2.radius()]),
            x_ref_dynamic_obst_0_1_leaf=obst1_position,
            xdot_ref_dynamic_obst_0_1_leaf=obst1_velocity,
            xddot_ref_dynamic_obst_0_1_leaf=obst1_acceleration,
            x_ref_dynamic_obst_1_1_leaf=obst2_position,
            xdot_ref_dynamic_obst_1_1_leaf=obst2_velocity,
            xddot_ref_dynamic_obst_1_1_leaf=obst2_acceleration,
            radius_body_1=np.array([0.1]),
        )
        ob, *_ = env.step(action)
    return {}


if __name__ == "__main__":
    arguments = sys.argv
    if len(arguments) == 1 or arguments[0] == 'dynamic_fabric':
        dynamic_fabric = True
    else:
        dynamic_fabric = False
    res = run_point_robot_example(n_steps=5000, dynamic_fabric=dynamic_fabric)
