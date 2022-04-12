import gym
import planarenvs.point_robot
from planarenvs.point_robot.envs.acc import PointRobotAccEnv
from planarenvs.sensors.goal_sensor import GoalSensor

from MotionPlanningGoal.staticSubGoal import StaticSubGoal



import numpy as np
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

# TODO: Currently, the goal sensor is restricted to observations up to the
# limits. Therefore, the robot cannot reach the actual goal which exceeds this
# limit <04-04-22, mspahn> #


def point_mass(n_steps=5000, render=True):
    """ Optional reconfiguration of the planner
    base_inertia = 0.3
    attractor_potential = "ca.norm_2(x_goal)**4"
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
    planner.set_components(fks, fk)
    planner.concretize()
    # Definition of the goal.
    goal_dict = {
        "m": 2,
        "w": 1.0,
        "prime": True,
        "indices": [0, 1],
        "parent_link": 0,
        "child_link": 1,
        "desired_position": [2.0, 1.0],
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
    # Start the simulation
    print("Starting simulation")
    goal_position = np.array(goal.position())
    for _ in range(n_steps):
        action = planner.compute_action(
            q=ob["x"], qdot=ob["xdot"], x_goal=goal_position
        )
        ob, *_ = env.step(action)
        goal_position = ob["GoalPosition"][0]
    return {}


if __name__ == "__main__":
    res = point_mass(n_steps=5000)
