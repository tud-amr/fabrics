import os
import gymnasium as gym
import numpy as np

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor

from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle

from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner
from fabrics.helpers.functions import get_rotation_matrix

absolute_path = os.path.dirname(os.path.abspath(__file__))
URDF_FILE = absolute_path + "/panda_dual_vacuum.urdf"


def initalize_environment(render=True):
    """
    Initializes the simulation environment.

    Adds obstacles and goal visualizaion to the environment based and
    steps the simulation once.
    """
    robots = [
        GenericUrdfReacher(urdf=URDF_FILE, mode="acc"),
    ]
    env: UrdfEnv  = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    full_sensor = FullSensor(
            goal_mask=["position", "weight"],
            obstacle_mask=['position', 'size'],
            variance=0.0
    )
    # Definition of the obstacle.
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [1000.5, -0.3, 0.3], "radius": 0.1},
    }
    obst1 = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [-1000.7, 0.0, 0.5], "radius": 0.1},
    }
    obst2 = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
    # Definition of the goal.
    angle = np.pi/2 * 2
    rot_matrix = get_rotation_matrix(angle, axis='z')
    goal_1 = np.array([0, 0.055, 0])
    goal_2 = np.array([0.185, 0, 0])
    goal_1 = np.dot(rot_matrix, goal_1)
    goal_2 = np.dot(rot_matrix, goal_2)
    goal_dict = {
        "subgoal0": {
            "weight": 2.0,
            "is_primary_goal": True,
            "indices": [0, 1, 2],
            "parent_link": "panda_link0",
            "child_link": "vacuum1_link",
            "desired_position": [0.1, -0.6, 0.4],
            "epsilon": 0.05,
            "type": "staticSubGoal",
        },
        "subgoal1": {
            "weight": 5,
            "is_primary_goal": False,
            "indices": [0, 1, 2],
            "parent_link": "vacuum1_link",
            "child_link": "vacuum2_link",
            "desired_position": goal_1.tolist(),
            "epsilon": 0.05,
            "type": "staticSubGoal",
        },
        "subgoal2": {
            "weight": 5,
            "is_primary_goal": False,
            "indices": [0, 1, 2],
            "parent_link": "panda_link7",
            "child_link": "vacuum_support_link",
            "desired_position": goal_2.tolist(),
            "epsilon": 0.05,
            "type": "staticSubGoal",
        }
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    obstacles = (obst1, obst2)
    env.reset()
    env.add_sensor(full_sensor, [0])
    for obst in obstacles:
        env.add_obstacle(obst)
    for sub_goal in goal.sub_goals():
        env.add_goal(sub_goal)
    env.set_spaces()
    return (env, goal)


def set_planner(goal: GoalComposition, degrees_of_freedom: int = 7):
    """
    Initializes the fabric planner for the panda robot.

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
    #     forward_kinematics,
    #     base_inertia=base_inertia,
    #     attractor_potential=attractor_potential,
    #     damper=damper,
    # )
    with open(URDF_FILE, "r", encoding="utf-8") as file:
        urdf = file.read()
    forward_kinematics = GenericURDFFk(
        urdf,
        rootLink="panda_link0",
        end_link=["vacuum1_link", "vacuum2_link", "vacuum_support_link"],
    )
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        forward_kinematics,
    )
    collision_links = ['panda_link7', 'panda_link3', 'panda_link4']
    panda_limits = [
            [-2.8973, 2.8973],
            [-1.7628, 1.7628],
            [-2.8973, 2.8973],
            [-3.0718, -0.0698],
            [-2.8973, 2.8973],
            [-0.0175, 3.7525],
            [-2.8973, 2.8973]
        ]
    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        collision_links=collision_links,
        goal=goal,
        number_obstacles=0,
        number_plane_constraints=0,
        #limits=panda_limits,
    )
    planner.concretize()
    return planner


def run_panda_example(n_steps=5000, render=True):
    (env, goal) = initalize_environment(render)
    planner = set_planner(goal)
    goal_2_leaf = planner.get_leaves(["goal_2_leaf"])[0]
    goal_1_leaf = planner.get_leaves(["goal_1_leaf"])[0]
    goal_2_leaf.concretize()
    goal_1_leaf.concretize()
    # planner.export_as_c("planner.c")
    action = np.zeros(7)
    ob, *_ = env.step(action)
    env.add_collision_link(0, 3, shape_type='sphere', size=[0.10])
    env.add_collision_link(0, 4, shape_type='sphere', size=[0.10])
    env.add_collision_link(0, 7, shape_type='sphere', size=[0.10])


    yaw_1 = np.pi/4 * 1
    yaw_2 = np.pi/2 * 0
    sub_goal_1_rotation_matrix = get_rotation_matrix(yaw_1, axis='z') #z->z 
    sub_goal_2_rotation_matrix = get_rotation_matrix(yaw_2, axis='y') #y->y
    for _ in range(n_steps):
        ob_robot = ob['robot_0']
        args = dict(
            q=ob_robot["joint_state"]["position"],
            qdot=ob_robot["joint_state"]["velocity"],
            x_goal_0=ob_robot['FullSensor']['goals'][4]['position'],
            weight_goal_0=ob_robot['FullSensor']['goals'][4]['weight'],
            x_goal_1=ob_robot['FullSensor']['goals'][5]['position'],
            weight_goal_1=ob_robot['FullSensor']['goals'][5]['weight'],
            x_goal_2=ob_robot['FullSensor']['goals'][6]['position'],
            weight_goal_2=ob_robot['FullSensor']['goals'][6]['weight'],
            x_obst_0=ob_robot['FullSensor']['obstacles'][2]['position'],
            radius_obst_0=ob_robot['FullSensor']['obstacles'][2]['size'],
            x_obst_1=ob_robot['FullSensor']['obstacles'][3]['position'],
            radius_obst_1=ob_robot['FullSensor']['obstacles'][3]['size'],
            #radius_body_links={3: 0.1, 4: 0.1, 7: 0.1},
            constraint_0=np.array([0, 0, 1, 0.0]),
        )
        action = planner.compute_action(**args)
        ob, *_ = env.step(action)
    env.close()
    return {}


if __name__ == "__main__":
    res = run_panda_example(n_steps=5000, render=True)
