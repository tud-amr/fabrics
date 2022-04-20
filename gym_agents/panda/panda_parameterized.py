import gym
import urdfenvs.panda_reacher

from MotionPlanningGoal.staticSubGoal import StaticSubGoal
from MotionPlanningEnv.sphereObstacle import SphereObstacle
from forwardkinematics.urdfFks.pandaFk import PandaFk



import numpy as np
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

# TODO: Currently, the goal sensor is restricted to observations up to the
# limits. Therefore, the robot cannot reach the actual goal which exceeds this
# limit <04-04-22, mspahn> #


def panda_parameterized(n_steps=5000, render=True):
    n = 7
    """ Optional reconfiguration of the planner """
    """
    base_inertia = 0.03
    attractor_potential = "20 * ca.norm_2(x)**4"
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
    panda_fk = PandaFk()
    fks = []
    for i in range(1, n):
        fks.append(panda_fk.fk(q, i, positionOnly=True))
    fk_ee = panda_fk.fk(q, n, positionOnly=True)
    fks.append(fk_ee)
    fk = fk_ee
    # The planner hides all the logic behind the function set_components.
    planner.set_components(fks, fk, number_obstacles = 2, goal = True)
    planner.concretize()
    # Definition of the obstacle.
    staticObstDict = {
        "dim": 3,
        "type": "sphere",
        "geometry": {"position": [0.3, -0.5, 0.3], "radius": 0.1},
    }
    obst1 = SphereObstacle(name="staticObst", contentDict=staticObstDict)
    staticObstDict = {
        "dim": 3,
        "type": "sphere",
        "geometry": {"position": [-0.7, 0.0, 0.5], "radius": 0.1},
    }
    obst2 = SphereObstacle(name="staticObst", contentDict=staticObstDict)
    # Definition of the goal.
    goal_dict = {
        "m": 3,
        "w": 1.0,
        "prime": True,
        "indices": [0, 1, 2],
        "parent_link": 0,
        "child_link": 7,
        "desired_position": [0.0, -0.8, 0.2],
        "epsilon": 0.05,
        "type": "staticSubGoal",
    }
    goal = StaticSubGoal(name="goal", contentDict=goal_dict)
    # Create the simulation
    env = gym.make("panda-reacher-acc-v0", dt=0.05, render=render)
    ob = env.reset()
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
            radius_body = np.array([0.02]),
        )
        ob, *_ = env.step(action)
    return {}


if __name__ == "__main__":
    res = panda_parameterized(n_steps=5000)
