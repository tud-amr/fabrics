import sys
import threading
import time
from tqdm import tqdm
from typing import Tuple, List
import numpy as np
from forwardkinematics.planarFks.point_fk import PointFk
from mpscenes.goals.goal_composition import GoalComposition
from planarenvs.point_robot.envs.acc import PointRobotAccEnv
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner
from mpscenes.obstacles.sphere_obstacle import SphereObstacle


DT = 0.01


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
) -> Tuple[PointRobotAccEnv,GoalComposition,List[SphereObstacle]]:
    #env = gym.make("point-robot-vel-v0", render=render, dt=0.01)
    env = PointRobotAccEnv(dt=DT, render=render)
    init_pos = np.array([-4.0, 0.8])
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
            "geometry": {"position": [-1.1, 0.8], "radius": 0.4},
    }
    obstacle_1 = SphereObstacle(name="obstacle_1", content_dict=static_obst_dict)
    static_obst_dict = {
            "type": "sphere",
            "geometry": {"position": [-0.3, -1.0], "radius": 0.6},
    }
    obstacle_2 = SphereObstacle(name="obstacle_1", content_dict=static_obst_dict)
    static_obst_dict = {
            "type": "sphere",
            "geometry": {"position": [1.2, 0.2], "radius": 0.4},
    }
    obstacle_3 = SphereObstacle(name="obstacle_1", content_dict=static_obst_dict)

    env.add_goal(goal)
    env.add_obstacle(obstacle_1)
    env.add_obstacle(obstacle_2)
    env.add_obstacle(obstacle_3)

    return (env, goal, [obstacle_1, obstacle_2, obstacle_3])

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
    collision_geometry = "-100/(1+ca.exp(5 * x - 3)) * (1 - ca.heaviside(xdot)) * xdot ** 2"
    collision_finsler = "100/(1+ca.exp(8 * x - 3)) * (1 - ca.heaviside(xdot)) * xdot ** 2"

    attractor_potential: str = "1.0 * ca.dot(x, ca.mtimes(np.identity(2), x))"
    attractor_metric: str = "ca.SX(np.identity(x.size()[0]))"
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        forward_kinematics,
        collision_geometry=collision_geometry,
        collision_finsler=collision_finsler,
        attractor_potential=attractor_potential,
        attractor_metric=attractor_metric,
        forcing_type = "simply_damped",
        damper_beta = "5.0",
    )
    collision_links = [3]
    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        collision_links=collision_links,
        goal=goal,
        number_obstacles=3,
    )
    planner.concretize()
    return planner

def run_planar_point(n_steps = 1000, render: bool = False):

    (env, goal, obstacles) = initialize_environment(render=render)
    obstacle = obstacles[0]
    obstacle_2 = obstacles[1]
    obstacle_3 = obstacles[2]
    planner = set_planner(goal)

    action = np.array([0.0, 0.0])
    ob, *_ = env.step(action)
    vels = []
    for i in tqdm(range(n_steps)):

        q = ob['joint_state']['position']
        qdot = ob['joint_state']['velocity']
        arguments = dict(
            q=q,
            qdot=qdot,
            x_goal_0=goal.sub_goals()[0].position(),
            weight_goal_0=goal.sub_goals()[0].weight(),
            x_obst_0=obstacle.position(),
            radius_obst_0=obstacle.radius(),
            x_obst_1=obstacle_2.position(),
            radius_obst_1=obstacle_2.radius(),
            x_obst_2=obstacle_3.position(),
            radius_obst_2=obstacle_3.radius(),
        )
        vels.append(np.linalg.norm(qdot))
        action = planner.compute_action(**arguments)
        #action = np.zeros(2)
        ob, _, done, info = env.step(action)
        if i > 100 and (done or np.linalg.norm(qdot) < 1e-4):
            print(info)
            break
    return {}


if __name__ == "__main__":
    render = True if sys.argv[1] == 'render' else False
    run_planar_point(n_steps=10000, render=render)
