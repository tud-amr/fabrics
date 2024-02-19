from copy import deepcopy
import os
import shutil
import gymnasium as gym
import numpy as np

from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk
from robotmodels.utils.robotmodel import RobotModel, LocalRobotModel


from urdfenvs.generic_mujoco.generic_mujoco_env import GenericMujocoEnv
from urdfenvs.generic_mujoco.generic_mujoco_robot import GenericMujocoRobot


from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle

from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

ROBOTTYPE = 'panda'
ROBOTMODEL = 'panda_without_gripper'


def initalize_environment(render=True):
    """
    Initializes the simulation environment.

    Adds obstacles and goal visualizaion to the environment based and
    steps the simulation once.
    """
    """
    robots = [
        GenericUrdfReacher(urdf="panda.urdf", mode="acc"),
    ]
    env: UrdfEnv  = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    """

    if os.path.exists(ROBOTTYPE):
        shutil.rmtree(ROBOTTYPE)
    robot_model = RobotModel(ROBOTTYPE, ROBOTMODEL)
    robot_model.copy_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), ROBOTTYPE))
    del robot_model

    robot_model = LocalRobotModel(ROBOTTYPE, ROBOTMODEL)


    xml_file = robot_model.get_xml_path()
    robots  = [
        GenericMujocoRobot(xml_file=xml_file, mode="vel"),
    ]
    home_config = np.array([-1.0,0,0,-1.57079,0,1.57079,-0.7853, 0.04, 0.04])



    # Definition of the obstacle.
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [0.5, 0.2, 0.2], "radius": 0.2},
    }
    obst1 = SphereObstacle(name="obstacle_1", content_dict=static_obst_dict)
    # Definition of the goal.
    goal_dict = {
        "subgoal0": {
            "weight": 1.0,
            "is_primary_goal": True,
            "indices": [0, 1, 2],
            "parent_link": "panda_link0",
            "child_link": "panda_hand",
            "desired_position": [0.5, 0.6, 0.3],
            "epsilon": 0.05,
            "type": "staticSubGoal",
        },
        "subgoal1": {
            "weight": 2.0,
            "is_primary_goal": False,
            "indices": [0, 1, 2],
            "parent_link": "panda_link7",
            "child_link": "panda_hand",
            "desired_position": [0.0, 0.0, -0.1],
            "epsilon": 0.05,
            "type": "staticSubGoal",
        }
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    obstacles = [obst1]
    env = GenericMujocoEnv(robots, obstacles, goal.sub_goals()[0:1], render=render)
    env.reset(pos=home_config)

    """
    env.add_sensor(full_sensor, [0])
    for obst in obstacles:
        env.add_obstacle(obst)
    for sub_goal in goal.sub_goals():
        env.add_goal(sub_goal)
    env.set_spaces()
    """
    return (env, goal, obstacles)


def set_planner(goal: GoalComposition, dt: float):
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
    collision_geometry: str = "-4.5 / (x ** 1) * (-0.5 * (ca.sign(xdot) - 1)) * xdot ** 2"
    geometry_plane_constraint: str = (
        "-10.0 / (x ** 1) * (-0.5 * (ca.sign(xdot) - 1)) * xdot ** 2"
    )
    finsler_plane_constraint: str = (
        "1.0/(x**1) * xdot**2"
    )
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    with open(absolute_path + "/panda_for_fk.urdf", "r", encoding="utf-8") as file:
        urdf = file.read()
    forward_kinematics = GenericURDFFk(
        urdf,
        rootLink="panda_link0",
        end_link="panda_link9",
    )
    planner = ParameterizedFabricPlanner(
        7,
        forward_kinematics,
        collision_geometry=collision_geometry,
        geometry_plane_constraint=geometry_plane_constraint,
    )
    collision_links = ['panda_link9']
    self_collision_pairs = {}
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
        self_collision_pairs=self_collision_pairs,
        goal=goal,
        number_obstacles=1,
        number_plane_constraints=1,
        limits=panda_limits,
    )
    planner.concretize(mode="vel", time_step=dt)
    return planner

def forward_simulate(
        planner: ParameterizedFabricPlanner,
        q0: np.ndarray,
        arguments: dict,
        n_steps: int,
        dt: float,
        ) -> np.ndarray:
    trajectory = np.zeros((n_steps, q0.size))
    dq = np.zeros_like(q0)
    q = deepcopy(q0)
    alpha = 0.43
    alpha = 0
    for i in range(n_steps):
        trajectory[i] = deepcopy(q)
        dq_new = planner.compute_action(q=q, qdot=dq, **arguments)
        dq = alpha * dq +  (1-alpha) * dq_new
        q = q + dq * dt

    return trajectory




def run_panda_example(n_steps=5000, render=True):
    (env, goal, obstacles) = initalize_environment(render)
    planner = set_planner(goal, env.dt)
    # planner.export_as_c("planner.c")
    action = np.zeros(7)
    ob, *_ = env.step(action)
    """
    body_links={1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 7: 0.1}
    for body_link, radius in body_links.items():
        env.add_collision_link(0, body_link, shape_type='sphere', size=[radius])
    """
    arguments = dict(
        x_goal_0=goal.sub_goals()[0].position(),
        weight_goal_0=goal.sub_goals()[0].weight(),
        x_goal_1=goal.sub_goals()[1].position(),
        weight_goal_1=goal.sub_goals()[1].weight(),
        radius_body_panda_link9=0.20,
        radius_obst_0=obstacles[0].size()[0],
        x_obst_0=obstacles[0].position(),
        constraint_0=np.array([0.0, 0.0, 1.0, -0.1]),
        #constraint_1=np.array([0.0, 1.0, 0.0, 0.0]),
    )

    q0 = ob['robot_0']['joint_state']['position'][0:7]
    trajectory_forward = forward_simulate(planner, q0, arguments, n_steps, env.dt)
    trajectory_actual = np.zeros_like(trajectory_forward)


    for i in range(n_steps):
        q = ob['robot_0']['joint_state']['position'][0:7]
        trajectory_actual[i] = q
        qdot = ob['robot_0']['joint_state']['velocity'][0:7]
        action = planner.compute_action(q=q, qdot=qdot, **arguments)
        ob, reward, terminated, truncated, info = env.step(action)

        """
        if terminated or truncated:
            print(info)
            break
        """
    env.close()
    return {}


if __name__ == "__main__":
    res = run_panda_example(n_steps=500, render=True)
