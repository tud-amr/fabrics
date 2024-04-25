import os
import time
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import quaternionic
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor

from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.obstacles.box_obstacle import BoxObstacle
from robotmodels.utils.robotmodel import RobotModel, LocalRobotModel
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner
from fabrics.helpers.translation import c2np

robot_model = RobotModel('dingo_kinova', model_name='dingo_kinova')
URDF_FILE = robot_model.get_urdf_path()

def initalize_environment(render=True, nr_obst: int = 0, n_cube_obst:int = 3):
    """
    Initializes the simulation environment.

    Adds obstacles and goal visualizaion to the environment based and
    steps the simulation once.
    """
    # robot_model = RobotModel('kinova', model_name='gen3_6dof')
    urdf_file = URDF_FILE
    robots = [
        GenericUrdfReacher(urdf=urdf_file, mode="acc"),
    ]
    env: UrdfEnv = UrdfEnv(
        robots=robots,
        dt=0.01,
        render=render,
        observation_checking=False,
    )
    full_sensor = FullSensor(
            goal_mask=["position", "weight"],
            obstacle_mask=['position', 'size'],
            variance=0.0
    )
    radius_ring = 0.3
    obstacles = []
    obstacle_resolution = n_cube_obst -1
    goal_orientation = [-0.366, 0.0, 0.0, 0.3305]
    rotation_matrix = quaternionic.array(goal_orientation).to_rotation_matrix
    whole_position = [0.1, 0.6, 0.8]
    for i in range(obstacle_resolution):
        angle = i/obstacle_resolution * 2.*np.pi
        origin_position = [
            0.0,
            radius_ring * np.cos(angle),
            radius_ring * np.sin(angle),
        ]
        position = np.dot(np.transpose(rotation_matrix), origin_position) + whole_position
        static_obst_dict = {
            "type": "box",
            "geometry": {"position": position.tolist(), "length": 0.1, "width": 0.1, "height": 0.1},
        }
        obstacles.append(BoxObstacle(name="staticObst", content_dict=static_obst_dict))

    static_obst_dict = {
            "type": "box",
            "geometry": {"position": [0.0, 0.0, 0.1], "length": 0.4, "width": 0.4, "height": 0.2},
        }
    obstacles.append(BoxObstacle(name="staticObst", content_dict=static_obst_dict))

    goal_dict = {
        "subgoal0": {
            "weight": 1.0,
            "is_primary_goal": True,
            "indices": [0, 1, 2],
            "parent_link": "world",
            "child_link": "arm_tool_frame",
            "desired_position": whole_position,
            "epsilon": 0.05,
            "type": "staticSubGoal",
        },
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    
    pos0 = np.array([-1.0, -1.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    env.reset(pos=pos0)
    env.add_sensor(full_sensor, [0])
    for obst in obstacles:
        env.add_obstacle(obst)
    for sub_goal in goal.sub_goals():
        env.add_goal(sub_goal)
    env.set_spaces()
    collision_radii = {3:0.35, 12: 0.1, 13:0.1, 14:0.1, 15: 0.1, 16: 0.1, 17: 0.1}
    for collision_link_nr in collision_radii.keys():
         env.add_collision_link(0, collision_link_nr, shape_type='sphere', size=[collision_radii[collision_link_nr]])
    return (env, goal)

def set_planner(goal: GoalComposition, n_obst: int, n_cube_obst:int, degrees_of_freedom: int):
    """
    Initializes the fabric planner for the kuka robot.

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
    # robot_model = RobotModel('kinova', model_name='gen3_6dof')
    urdf_file = URDF_FILE
    with open(urdf_file, "r", encoding="utf-8") as file:
        urdf = file.read()
    forward_kinematics = GenericURDFFk(
        urdf,
        rootLink="world",
        end_link="arm_tool_frame",
    )
    planner = ParameterizedFabricPlanner(
        degrees_of_freedom,
        forward_kinematics,
    )
    collision_links = [
        "base_link_y",
        "arm_arm_link",
        "arm_forearm_link",
        "arm_lower_wrist_link",
        "arm_upper_wrist_link",
        "arm_end_effector_link"
    ]
    # 3 omnibase joints + 6 arm joints
    omnibase_limits = np.array([
        [-4, 4],
        [-4, 4],
        [-np.pi, np.pi]])
    kinova_gen3lite_limits = np.array([
        [-154.1, 154.1],
        [150.1, 150.1],
        [150.1, 150.1],
        [-148.98, 148.98],
        [-144.97, 145.0],
        [-148.98, 148.98]]) * np.pi/180
    joint_limits = list(np.concatenate((omnibase_limits, kinova_gen3lite_limits)))

    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        collision_links=collision_links,
        goal=goal,
        number_obstacles=n_obst,
        number_obstacles_cuboid=n_cube_obst,
        number_plane_constraints=0,
        limits=joint_limits,
    )
    planner.concretize()
    return planner

def run_kinova_example(n_steps=5000, render=True, dof=3+6):
    comp_time = []
    Ns = []
    nr_obst = 0
    n_cube_obst = 3
    total_obst = nr_obst + n_cube_obst
    (env, goal) = initalize_environment(render, nr_obst=nr_obst, n_cube_obst=n_cube_obst)
    planner = set_planner(goal, n_obst=nr_obst, n_cube_obst=n_cube_obst, degrees_of_freedom=dof)
    planner.export_as_c('planner.c')
    python_code = c2np('planner.c', 'generated_code/planner_np.py' )
    from examples.generated_code.planner_np import casadi_f0_numpy

    # inputs from fabrics/helpers/casadiFunctionWrapper.py (they are in alphabetical order)

    # 'q': SX([q_0, q_1, q_2, q_3, q_4, q_5, q_6, q_7, q_8]),
    # 'qdot': SX([qdot_0, qdot_1, qdot_2, qdot_3, qdot_4, qdot_5, qdot_6, qdot_7, qdot_8]),
    # 'radius_body_arm_arm_link': SX(radius_body_arm_arm_link),
    # 'radius_body_arm_end_effector_link': SX(radius_body_arm_end_effector_link),
    # 'radius_body_arm_forearm_link': SX(radius_body_arm_forearm_link),
    # 'radius_body_arm_lower_wrist_link': SX(radius_body_arm_lower_wrist_link),
    # 'radius_body_arm_upper_wrist_link': SX(radius_body_arm_upper_wrist_link),
    # 'radius_body_base_link_y': SX(radius_body_base_link_y),
    # 'size_obst_cuboid_0': SX([size_obst_cuboid_0_0, size_obst_cuboid_0_1, size_obst_cuboid_0_2]),
    # 'size_obst_cuboid_1': SX([size_obst_cuboid_1_0, size_obst_cuboid_1_1, size_obst_cuboid_1_2]),
    # 'size_obst_cuboid_2': SX([size_obst_cuboid_2_0, size_obst_cuboid_2_1, size_obst_cuboid_2_2]),
    # 'weight_goal_0': SX(weight_goal_0),
    # 'x_goal_0': SX([x_goal_0_0, x_goal_0_1, x_goal_0_2]),
    # 'x_obst_cuboid_0': SX([x_obst_cuboid_0_0, x_obst_cuboid_0_1, x_obst_cuboid_0_2]),
    # 'x_obst_cuboid_1': SX([x_obst_cuboid_1_0, x_obst_cuboid_1_1, x_obst_cuboid_1_2]),
    # 'x_obst_cuboid_2': SX([x_obst_cuboid_2_0, x_obst_cuboid_2_1, x_obst_cuboid_2_2]),

    action = np.zeros(dof)
    ob, *_ = env.step(action)
    
    for w in tqdm.tqdm(range(n_steps)):
        t0 = time.perf_counter()
        N = int((w+1)*2)
        ob_robot = ob['robot_0']
        #  x_obsts_cuboid is a np.array of shape (n_cube_obst, 3)
        #  size_obsts_cuboid is a np.array of shape (n_cube_obst, 3), 3: length", "width", "height ?
        x_obsts = [ob_robot['FullSensor']['obstacles'][i+2]['position'] for i in range(n_cube_obst)]
        size_obsts = [ob_robot['FullSensor']['obstacles'][i+2]['size'] for i in range(n_cube_obst)]

        # Variables that do not change
        radius_body_arm_arm_link_n = np.repeat(np.array([0.1]), N)
        radius_body_arm_end_effector_link_n = np.repeat(np.array([0.1]), N)
        radius_body_arm_forearm_link_n = np.repeat(np.array([0.1]), N)
        radius_body_arm_lower_wrist_link_n = np.repeat(np.array([0.1]), N)
        radius_body_arm_upper_wrist_link_n = np.repeat(np.array([0.1]), N)
        radius_body_base_link_y_n = np.repeat(np.array([0.35]), N)
        size_obsts_cuboid_0_n = np.repeat(size_obsts[0].reshape(-1,1), N, axis=1)
        size_obsts_cuboid_1_n = np.repeat(size_obsts[1].reshape(-1,1), N, axis=1)
        size_obsts_cuboid_2_n = np.repeat(size_obsts[2].reshape(-1,1), N, axis=1)
        weight_goal_0_n = np.repeat(np.array([ob_robot['FullSensor']['goals'][total_obst+2]['weight']]), N)

        #  Variables that may change:.repeat in columns
        q_n = np.repeat(ob_robot["joint_state"]["position"].reshape(-1,1), N, axis=1)
        qdot_n = np.repeat(ob_robot["joint_state"]["velocity"].reshape(-1,1), N, axis=1)

        x_goal_0_n = np.repeat(ob_robot['FullSensor']['goals'][total_obst+2]['position'].reshape(-1,1), N, axis=1)
        x_obst_cuboid_0_n = np.repeat(x_obsts[0].reshape(-1,1), N, axis=1)
        x_obst_cuboid_1_n = np.repeat(x_obsts[1].reshape(-1,1), N, axis=1)
        x_obst_cuboid_2_n = np.repeat(x_obsts[2].reshape(-1,1), N, axis=1)

        t1 = time.perf_counter()
        action_n = casadi_f0_numpy(
            q_n,
            qdot_n,
            radius_body_arm_arm_link_n,
            radius_body_arm_end_effector_link_n,
            radius_body_arm_forearm_link_n,
            radius_body_arm_lower_wrist_link_n,
            radius_body_arm_upper_wrist_link_n,
            radius_body_base_link_y_n,
            size_obsts_cuboid_0_n,
            size_obsts_cuboid_1_n,
            size_obsts_cuboid_2_n,
            weight_goal_0_n,
            x_goal_0_n,
            x_obst_cuboid_0_n,
            x_obst_cuboid_1_n,
            x_obst_cuboid_2_n
        )
        t2 = time.perf_counter()
        comp_time.append(t2-t1)
        Ns.append(N)
        ob, *_ = env.step(action_n[0])
    
    env.close()
    return {"comp_time": comp_time, "Ns": Ns}


if __name__ == "__main__":
    res = run_kinova_example(n_steps=250, render=False, dof=9)

    mean_single_env_time = 0.27/1000
    comp_time = np.array(res["comp_time"])
    Ns = np.array(res["Ns"])
    comp_time_loop = Ns * mean_single_env_time

    #  plot the computation time as a function of Ns
    average_time_per_N = comp_time/Ns
    fig, ax = plt.subplots()
    ax.plot(Ns, comp_time*1000)
    ax.plot(Ns, comp_time_loop*1000, 'r--')
    ax.set_xlabel("N in parallel")
    ax.set_ylabel("TOTAL computation time (ms)")
    ax.legend(["parallel computation (numpy)", "approx loop computation (casadi)"])
    ax.grid()
    ax.set_title("Computation using Numpy arrays")
    fig.savefig("images/computation_time.png")

    # plot the average computation time per N
    fig, ax = plt.subplots()
    ax.plot(Ns, average_time_per_N*1000)
    ax.axhline(y=mean_single_env_time*1000, color='r', linestyle='--')
    ax.set_xlabel("N in parallel")
    ax.set_ylabel("average computation time per N (ms)")
    ax.set_yscale('log')
    ax.grid(True, which="both", linestyle="-", linewidth=0.5)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle="--", linewidth=0.5)
    ax.set_title("Computation using Numpy arrays")
    ax.legend(["average computation time per N", "single avg computation time casadi (ms)"])
    fig.savefig("images/average_computation_time_per_N.png")

