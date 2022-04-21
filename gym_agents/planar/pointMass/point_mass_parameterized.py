import gym
import planarenvs.point_robot
from planarenvs.point_robot.envs.acc import PointRobotAccEnv
from planarenvs.sensors.goal_sensor import GoalSensor

from MotionPlanningGoal.staticSubGoal import StaticSubGoal
from MotionPlanningEnv.sphereObstacle import SphereObstacle



import numpy as np
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

# TODO: Currently, the goal sensor is restricted to observations up to the
# limits. Therefore, the robot cannot reach the actual goal which exceeds this
# limit <04-04-22, mspahn> #


def point_mass_parameterized(n_steps=5000, render=True):
    """ Optional reconfiguration of the planner """
    """
    base_inertia = 0.1
    attractor_potential = "1 * ca.norm_2(x)**2"
    damper = {
        "alpha_b": 0.5,
        "alpha_eta": 0.5,
        "alpha_shift": 0.5,
        "beta_distant": 0.01,
        "beta_close": 6.5,
        "radius_shift": 0.1,
    }
    planner = ParameterizedFabricPlanner(
        2,
        base_inertia=base_inertia,
        attractor_potential=attractor_potential,
        damper=damper,
    )
    """
    planner = ParameterizedFabricPlanner(2) # dof = 2
    fks = [planner.variables.position_variable()]
    fk = planner.variables.position_variable()
    # The planner hides all the logic behind the function set_components.
    planner.set_components(fks, fk, number_obstacles = 2)
    planner.concretize()
    # Definition of the obstacle.
    staticObstDict = {
        "dim": 2,
        "type": "sphere",
        "geometry": {"position": [-0.0, 3.0], "radius": 0.6},
    }
    obst1 = SphereObstacle(name="staticObst", contentDict=staticObstDict)
    staticObstDict = {
        "dim": 2,
        "type": "sphere",
        "geometry": {"position": [0.0, -1.0], "radius": 0.4},
    }
    obst2 = SphereObstacle(name="staticObst", contentDict=staticObstDict)
    # Definition of the goal.
    goal_dict = {
        "m": 2,
        "w": 1.0,
        "prime": True,
        "indices": [0, 1],
        "parent_link": 0,
        "child_link": 1,
        "desired_position": [-4.0, 1.0],
        "epsilon": 0.2,
        "type": "staticSubGoal",
    }
    goal = StaticSubGoal(name="goal", contentDict=goal_dict)
    # Create the simulation
    env: PointRobotAccEnv = gym.make(
        "point-robot-acc-v0", dt=0.01, render=render
    )
    x0 = np.array([4.3, -1.0])
    xdot0 = np.array([-1.0, 0.0])
    ob = env.reset(pos=x0, vel=xdot0)
    sensor = GoalSensor(nb_goals=1)
    env.add_sensor(sensor)
    env.add_goal(goal)
    env.add_obstacle(obst1)
    env.add_obstacle(obst2)
    # Start the simulation
    print("Starting simulation")
    goal_position = np.array(goal.position())
    obst1_position = np.array(obst1.position())
    obst2_position = np.array(obst2.position())
    for _ in range(n_steps):
        action = planner.compute_action(
            q=ob["x"], qdot=ob["xdot"],
            x_goal = goal_position,
            x_obst_0 = obst1_position,
            x_obst_1 = obst2_position,
            radius_obst_0 = np.array([obst1.radius()]),
            radius_obst_1 = np.array([obst2.radius()]),
            radius_body = np.array([0.1])
        )
        ob, *_ = env.step(action)
        goal_position = ob["GoalPosition"][0]
    return {}


if __name__ == "__main__":
    res = point_mass_parameterized(n_steps=5000)
