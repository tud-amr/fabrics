import sys
import threading
import time
from typing import Tuple
import numpy as np
from forwardkinematics.planarFks.point_fk import PointFk
from mpscenes.goals.goal_composition import GoalComposition
from planarenvs.point_robot.envs.acc import PointRobotAccEnv
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner
import matplotlib.pyplot as plt
from mpscenes.obstacles.sphere_obstacle import SphereObstacle


DT = 0.001


def plt_show_sec(duration: float = 3):
    def _stop():
        time.sleep(duration)
        plt.close()
    if duration:
        threading.Thread(target=_stop).start()
    plt.show(block=False)
    plt.pause(duration)



def initialize_environment(
    n_steps: int = 1000,
    render: bool = False,
) -> Tuple[PointRobotAccEnv,GoalComposition,]:
    #env = gym.make("point-robot-vel-v0", render=render, dt=0.01)
    env = PointRobotAccEnv(dt=DT, render=render)
    init_pos = np.array([-4.0, 0.1])
    init_vel = np.array([1.0, 0.0])
    ob = env.reset(pos=init_pos, vel=init_vel)
    env.reset_limits(
        pos={"high": np.array([5.0, 5.0]), "low": np.array([-5.0, -5.0])}
    )
    goal_dict = {
        "subgoal0": {
            "weight": 1.0,
            "is_primary_goal": True,
            "indices": [0, 1],
            "parent_link": 0,
            "child_link": 3,
            "desired_position": [3, -0.1],
            "epsilon": 0.02,
            "type": "staticSubGoal",
        }
    }
    goal = GoalComposition(name="goal1", content_dict=goal_dict)
    static_obst_dict = {
            "type": "sphere",
            "geometry": {"position": [0.3, 0.0], "radius": 1.0},
    }
    obstacle_1 = SphereObstacle(name="obstacle_1", content_dict=static_obst_dict)

    env.add_goal(goal)
    env.add_obstacle(obstacle_1)

    return (env, goal, obstacle_1)

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
    forward_kinematics = PointFk()
    collision_geometry = "-2.0 / (x ** 1) * xdot ** 2"
    collision_finsler = "1.0/(x**2) * (1 - ca.heaviside(xdot))* xdot**2"
    #collision_finsler = "1.0/(x**2) * xdot**2"
    attractor_potential: str = "0.5 * ca.dot(x, ca.mtimes(np.identity(2), x))"
    attractor_metric: str = "ca.SX(np.identity(x.size()[0]))"
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        forward_kinematics,
        collision_geometry=collision_geometry,
        collision_finsler=collision_finsler,
        attractor_potential=attractor_potential,
        attractor_metric=attractor_metric,
        forcing_type = "constantly_damped",
        damper_beta = "2.5",
    )
    collision_links = [3]
    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        collision_links=collision_links,
        goal=goal,
        number_obstacles=1,
    )
    planner.concretize()
    return planner

def run_planar_point(render: bool = False):

    (env, goal, obstacle) = initialize_environment(render=render)
    planner = set_planner(goal)

    action = np.array([0.0, 0.0])
    ob, *_ = env.step(action)
    vels = []
    T = 50
    n_steps = int(T/DT)
    for i in range(n_steps):

        q = ob['joint_state']['position']
        qdot = ob['joint_state']['velocity']
        arguments = dict(
            q=q,
            qdot=qdot,
            x_goal_0=goal.sub_goals()[0].position(),
            weight_goal_0=goal.sub_goals()[0].weight(),
            x_obst_0=obstacle.position(),
            radius_obst_0=obstacle.radius(),
        )
        vels.append(np.linalg.norm(qdot))
        action = planner.compute_action(**arguments)
        #action = np.zeros(2)
        ob, _, done, info = env.step(action)
        if i > 100 and (done or np.linalg.norm(qdot) < 1e-4):
            print(info)
            #break
    plt.plot(vels)
    plt_show_sec(2)


if __name__ == "__main__":
    render = True if sys.argv[1] == 'render' else False
    run_planar_point(render=render)
