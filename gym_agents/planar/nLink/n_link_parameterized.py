import gym
import planarenvs.n_link_reacher
from planarenvs.point_robot.envs.acc import PointRobotAccEnv
from planarenvs.sensors.goal_sensor import GoalSensor

from MotionPlanningGoal.staticSubGoal import StaticSubGoal
from MotionPlanningEnv.sphereObstacle import SphereObstacle
from forwardkinematics.planarFks.planarArmFk import PlanarArmFk



import numpy as np
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

# TODO: Currently, the goal sensor is restricted to observations up to the
# limits. Therefore, the robot cannot reach the actual goal which exceeds this
# limit <04-04-22, mspahn> #


def n_link_parameterized(n_steps=5000, n=3, render=True):
    """ Optional reconfiguration of the planner """
    """
    base_inertia = 0.3
    attractor_potential = "ca.norm_2(x)**4"
    damper = {
        "alpha_b": 0.5,
        "alpha_eta": 0.5,
        "alpha_shift": 0.5,
        "beta_distant": 0.01,
        "beta_close": 6.5,
        "radius_shift": 0.1,
    }
    planner = ParameterizedFabricPlanner(
        n,
        base_inertia=base_inertia,
        attractor_potential=attractor_potential,
        damper=damper,
    )
    """
    planner = ParameterizedFabricPlanner(n) # dof = n
    q = planner.variables.position_variable()
    planarArmFk = PlanarArmFk(n)
    fks = []
    for i in range(1, n):
        fks.append(planarArmFk.fk(q, i, positionOnly=True))
    fk_ee = planarArmFk.fk(q, n, positionOnly=True)
    fks.append(fk_ee)
    fk = fk_ee
    # The planner hides all the logic behind the function set_components.
    planner.set_components(fks, fk, number_obstacles = 2)
    planner.concretize()
    # Definition of the obstacle.
    staticObstDict = {
        "dim": 2,
        "type": "sphere",
        "geometry": {"position": [0.5, -0.5], "radius": 0.2},
    }
    obst1 = SphereObstacle(name="staticObst", contentDict=staticObstDict)
    staticObstDict = {
        "dim": 2,
        "type": "sphere",
        "geometry": {"position": [-1.0, -1.0], "radius": 0.1},
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
        "desired_position": [0.0, -1.0],
        "epsilon": 0.2,
        "type": "staticSubGoal",
    }
    goal = StaticSubGoal(name="goal", contentDict=goal_dict)
    # Create the simulation
    env = gym.make("nLink-reacher-acc-v0", n=n, dt=0.05, render=render)
    x0 = np.array([0.3, -1.0, 0.0])
    xdot0 = np.array([-0.0, 0.0, 0.0])
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
            x_obst_0 = obst2_position,
            x_obst_1 = obst1_position,
            radius_obst_0 = np.array([obst1.radius()]),
            radius_obst_1 = np.array([obst2.radius()]),
            radius_body = np.array([0.2]),
        )
        ob, *_ = env.step(action)
        goal_position = ob["GoalPosition"][0]
    return {}


if __name__ == "__main__":
    res = n_link_parameterized(n_steps=5000)
