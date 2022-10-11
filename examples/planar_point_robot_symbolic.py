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
        "type": "sphere",
        "geometry": {"position": [0.0, 1.1], "radius": 1.6},
    }
    obst1 = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [0.0, -1.0], "radius": 0.4},
    }
    obst2 = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
    # Definition of the goal.
    goal_dict = {
        "subgoal0": {
            "weight": 1.0,
            "is_primary_goal": True,
            "indices": [0, 1],
            "parent_link": 0,
            "child_link": 2,
            "desired_position": [-3.0, 2.0],
            "epsilon": 0.15,
            "type": "staticSubGoal",
        }
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
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
    collision_geometry: str = (
        "-sym('k_geo') / (x ** sym('exp_geo')) * (-0.5 * (ca.sign(xdot) - 1)) * xdot ** 2"
    )
    planner = ParameterizedFabricPlanner(
            degrees_of_freedom,
            robot_type,
            collision_geometry=collision_geometry
        )
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


def run_point_robot_symbolic(n_steps=5000, render=True):
    (env, obstacles, goal, initial_observation) = initalize_environment(
        render=render
    )
    ob = initial_observation
    obst0 = obstacles[0]
    obst1 = obstacles[1]
    planner = set_planner(goal)

    # Start the simulation
    print("Starting simulation")
    for _ in range(n_steps):
        action = planner.compute_action(
            q=ob["joint_state"]["position"],
            qdot=ob["joint_state"]["velocity"],
            x_goal_0=goal.sub_goals()[0].position(),
            weight_goal_0=goal.sub_goals()[0].weight(),
            x_obst_0=obst0.position(),
            x_obst_1=obst1.position(),
            radius_obst_0=obst0.radius(),
            radius_obst_1=obst1.radius(),
            radius_body_1=0.10,
            k_geo_obst_0_1_leaf=15.0,
            exp_geo_obst_0_1_leaf=1.0,
            k_geo_obst_1_1_leaf=15.0,
            exp_geo_obst_1_1_leaf=1.0,
        )
        ob, *_ = env.step(action)
    return {}


if __name__ == "__main__":
    res = run_point_robot_symbolic(n_steps=5000)
