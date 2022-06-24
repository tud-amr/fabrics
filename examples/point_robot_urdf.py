import gym
import numpy as np
import urdfenvs.point_robot_urdf
from MotionPlanningEnv.sphereObstacle import SphereObstacle
from MotionPlanningGoal.goalComposition import GoalComposition
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner
"""
Fabrics example for a 3D point mass robot.
The fabrics planner uses a 2D point mass to compute actions for a simulated 3D point mass.

To do: tune behavior.
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
    env = gym.make("pointRobotUrdf-acc-v0", dt=0.05, render=render)
    # Set the initial position and velocity of the point mass.
    pos0 = np.array([0.0, 0.0, 0.0])
    vel0 = np.array([0.0, 0.0, 0.0])
    initial_observation = env.reset(pos=pos0, vel=vel0)
    # Definition of the obstacle.
    static_obst_dict = {
            "dim": 3,
            "type": "sphere",
            "geometry": {"position": [2.0, 0.0, 0.0], "radius": 0.5},
    }
    obst1 = SphereObstacle(name="staticObst1", contentDict=static_obst_dict)
    obstacles = (obst1) # Add additional obstacles here.
    # Definition of the goal.
    goal_dict = {
            "subgoal0": {
                "m": 2,
                "w": 2,
                "prime": True,
                "indices": [0, 1],
                "parent_link" : 0,
                "child_link" : 1,
                "desired_position": [3.5, 0.0],
                "epsilon" : 0.1,
                "type": "staticSubGoal"
            }
    }
    goal = GoalComposition(name="goal", contentDict=goal_dict)
    # Add walls, the goal and the obstacle to the environment.
    env.set_walls(limits=[[-5, -5], [5, 5]])
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
    degrees_of_freedom = 2
    robot_type = "pointRobot"
    # Optional reconfiguration of the planner with collision_geometry/finsler, remove for defaults.
    collision_geometry = "-0.5 / (x ** 5) * (-0.5 * (ca.sign(xdot * 2) - 0.5)) * xdot ** 2"
    collision_finsler = "0.1/((x - 0.3)**1) * xdot**2"
    planner = ParameterizedFabricPlanner(
            degrees_of_freedom,
            robot_type,
            collision_geometry=collision_geometry,
            collision_finsler=collision_finsler
    )
    collision_links = [1]
    self_collision_links = {}
    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        collision_links,
        self_collision_links,
        goal,
        number_obstacles=1,
    )
    planner.concretize()
    return planner


def run_point_mass_example(n_steps=10000, render=True):
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
    action = np.array([0.0, 0.0, 0.0])
    planner = set_planner(goal)
    # Start the simulation.
    print("Starting simulation")
    sub_goal_0_position = np.array(goal.subGoals()[0].position())
    sub_goal_0_weight = np.array(goal.subGoals()[0].weight())
    obst1_position = np.array(obst1.position())
    for _ in range(n_steps):
        # Calculate action with the fabric planner, slice the states to drop Z-axis [3] information.
        action[0:2] = planner.compute_action(
            q=ob["x"][0:2],
            qdot=ob["xdot"][0:2],
            x_goal_0=sub_goal_0_position,
            weight_goal_0=sub_goal_0_weight,
            x_obst_0=obst1_position[0:2],
            radius_obst_0=np.array([obst1.radius()]),
            radius_body_1=np.array([0.2])
        )
        # One step forward using the action provided by the fabric planner, currently clipped to set minimum and maximum value.
        ob, *_, = env.step(np.clip(action, -2.174, 2.174))
    return {}


if __name__ == "__main__":
    res = run_point_mass_example(n_steps=10000)

