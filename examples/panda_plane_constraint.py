import gymnasium as gym
import os
import numpy as np
import quaternionic

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor

from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle

from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

# TODO: Angle cannot be read through the FullSensor

def initalize_environment(render=True, obstacle_resolution = 8):
    """
    Initializes the simulation environment.

    Adds obstacles and goal visualizaion to the environment based and
    steps the simulation once.
    """
    robots = [
        GenericUrdfReacher(urdf="panda.urdf", mode="acc"),
    ]
    env: UrdfEnv  = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    ).unwrapped
    full_sensor = FullSensor(
            goal_mask=["position", "weight"],
            obstacle_mask=["position", "size"],
            variance=0.0,
    )
    q0 = np.array([0.0, -1.0, 0.0, -1.501, 0.0, 1.8675, 0.0])
    # Definition of the obstacle.
    radius_ring = 0.3
    obstacles = []
    goal_orientation = [-0.366, 0.0, 0.0, 0.3305]
    rotation_matrix = quaternionic.array(goal_orientation).to_rotation_matrix
    whole_position = [0.1, 0.6, 0.8]
    for i in range(obstacle_resolution + 1):
        angle = i/obstacle_resolution * 2.*np.pi
        origin_position = [
            0.0,
            radius_ring * np.cos(angle),
            radius_ring * np.sin(angle),
        ]
        position = np.dot(np.transpose(rotation_matrix), origin_position) + whole_position
        static_obst_dict = {
            "type": "sphere",
            "geometry": {"position": position.tolist(), "radius": 0.1},
        }
        obstacles.append(SphereObstacle(name="staticObst", content_dict=static_obst_dict))
    # Definition of the goal.
    goal_dict = {
        "subgoal0": {
            "weight": 1.0,
            "is_primary_goal": True,
            "indices": [0, 1, 2],
            "parent_link": "panda_link0",
            "child_link": "panda_hand",
            "desired_position": whole_position,
            "epsilon": 0.05,
            "type": "staticSubGoal",
        },
        "subgoal1": {
            "weight": 3.0,
            "is_primary_goal": False,
            "indices": [0, 1, 2],
            "parent_link": "panda_link7",
            "child_link": "panda_hand",
            "desired_position": [0.107, 0.0, 0.0],
            "angle": goal_orientation,
            "epsilon": 0.05,
            "type": "staticSubGoal",
        }
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    env.reset(pos=q0)
    env.add_sensor(full_sensor, [0])
    for obst in obstacles:
        env.add_obstacle(obst)
    for sub_goal in goal.sub_goals():
        env.add_goal(sub_goal)
    env.set_spaces()
    return (env, goal)


def set_planner(goal: GoalComposition, degrees_of_freedom: int = 7, obstacle_resolution = 10):
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

    robot_type = 'panda'

    ## Optional reconfiguration of the planner
    # base_inertia = 0.03
    # attractor_potential = "5.0 * (ca.norm_2(x) + 1 /10 * ca.log(1 + ca.exp(-2 * 10 * ca.norm_2(x))))"
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
    # attractor_potential = "15.0 * (ca.norm_2(x) + 1 /10 * ca.log(1 + ca.exp(-2 * 10 * ca.norm_2(x))))"
    # collision_geometry= "-0.1 / (x ** 2) * (-0.5 * (ca.sign(xdot) - 1)) * xdot ** 2"
    # collision_finsler= "0.1/(x**1) * xdot**2"
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    with open(absolute_path + "/albert_polluted_2.urdf", "r", encoding='utf-8') as file:
        urdf = file.read()
    forward_kinematics = GenericURDFFk(
        urdf,
        rootLink="panda_link0",
        end_link=["panda_vacuum", "panda_vacuum_2"],
    )
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        forward_kinematics,
        geometry_plane_constraint="10*(1/(1+1*ca.exp(-10*x))-1) * (xdot**2)",
    )
    panda_limits = [
            [-2.8973, 2.8973],
            [-1.7628, 1.7628],
            [-2.8973, 2.8973],
            [-3.0718, -0.0698],
            [-2.8973, 2.8973],
            [-0.0175, 3.7525],
            [-2.8973, 2.8973]
        ]
    collision_links = ['panda_link1', 'panda_link4', 'panda_link6', 'panda_hand']
    self_collision_pairs = {}
    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        collision_links=collision_links,
        goal=goal,
        number_obstacles=obstacle_resolution,
        limits=panda_limits,
        number_plane_constraints=1,
    )
    planner.concretize()
    return planner


def run_panda_ring_example(n_steps=5000, render=True, serialize=False, planner=None):
    obstacle_resolution_ring = 10
    (env, goal) = initalize_environment(
        render=render,
        obstacle_resolution=obstacle_resolution_ring
    )
    action = np.zeros(7)
    ob, *_ = env.step(action)
    env.reconfigure_camera(1.4000000953674316, 67.9999008178711, -31.0001220703125, (-0.4589785635471344, 0.23635289072990417, 0.3541859984397888))

    constraint = np.array([0, 0, 1, 0.0])

    if not planner:
        planner = set_planner(goal, obstacle_resolution = obstacle_resolution_ring)
        # Serializing the planner is optional
        if serialize:
            planner.serialize('serialized_10.pbz2')

    sub_goal_0_quaternion = quaternionic.array(goal.sub_goals()[1].angle())
    sub_goal_0_rotation_matrix = sub_goal_0_quaternion.to_rotation_matrix

    for _ in range(n_steps):
        ob_robot = ob['robot_0']
        x_obsts = [
            ob_robot['FullSensor']['obstacles'][i+2]['position'] for i in range(obstacle_resolution_ring)
        ]
        radius_obsts = [
            ob_robot['FullSensor']['obstacles'][i+2]['size'] for i in range(obstacle_resolution_ring)
        ]
        action = planner.compute_action(
            q=ob_robot["joint_state"]["position"],
            qdot=ob_robot["joint_state"]["velocity"],
            x_obsts=x_obsts,
            radius_obsts=radius_obsts,
            x_goal_0=ob_robot['FullSensor']['goals'][obstacle_resolution_ring+3]['position'],
            weight_goal_0=ob_robot['FullSensor']['goals'][obstacle_resolution_ring+3]['weight'],
            x_goal_1=ob_robot['FullSensor']['goals'][obstacle_resolution_ring+4]['position'],
            weight_goal_1=ob_robot['FullSensor']['goals'][obstacle_resolution_ring+4]['weight'],
            radius_body_panda_link1=0.1,
            radius_body_panda_link4=0.1,
            radius_body_panda_link6=0.15,
            radius_body_panda_hand=0.1,
            angle_goal_1=np.array(sub_goal_0_rotation_matrix),
            constraint_0=constraint
        )
        ob, *_ = env.step(action)
    env.close()
    return {}


if __name__ == "__main__":
    res = run_panda_ring_example(n_steps=10000, serialize = True)
