import gym
import numpy as np
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.goals.goal_composition import GoalComposition
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner
import pybullet as p
import matplotlib.pyplot as plt
from scipy import ndimage

# Fabrics example for a 3D point mass robot. The fabrics planner uses a 2D point
# mass to compute actions for a simulated 3D point mass.
#
# todo: tune behavior.

def edf(pos, proj_rgb): #-> float:
    #to binary image, obstacles are red
    proj_r = proj_rgb[:, :, 1]
    proj_bin = ((1-proj_r) > 0.9)
    kkk=1
    plt.subplot(1, 3, 1)
    plt.imshow(proj_r)

    plt.subplot(1, 3, 2)
    plt.imshow(proj_bin)

    dist_map = ndimage.distance_transform_edt(1-proj_bin)
    dist_map = dist_map/5 # /100 pixels * 20 in meters

    # convert pos to pixels
    pos_p = [round((pos[0]+10)*5), round((-pos[1]+10)*5)]

    # dist_map = dist_map_t
    plt.subplot(1, 3, 3)
    plt.imshow(dist_map)
    plt.show()

    # index in map
    dist_pos = dist_map[pos_p[0], pos_p[1]]

    gradient_map = np.gradient(dist_map)
    gradient_x = gradient_map[0]
    gradient_y = gradient_map[1]
    grad_x_pos = gradient_x[pos_p[0], pos_p[1]]
    grad_y_pos = gradient_y[pos_p[0], pos_p[1]]
    k = 111
    return (dist_pos, grad_x_pos, grad_y_pos) #dist_map

def edf_jacobian(pos) -> float:


    #SARAYS FUNCTION

    return np.zeros(2)

def initalize_environment(render):
    """
    Initializes the simulation environment.

    Adds an obstacle and goal visualizaion to the environment and
    steps the simulation once.
    
    Params
    ----------
    render
        Boolean toggle to set rendering on (True) or off (False).
    """
    robots = [
        GenericUrdfReacher(urdf="pointRobot.urdf", mode="acc"),
    ]
    env: UrdfEnv  = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    # Set the initial position and velocity of the point mass.
    pos0 = np.array([-2.0, 0.5, 0.0])
    vel0 = np.array([0.1, 0.0, 0.0])
    full_sensor = FullSensor(goal_mask=["position"], obstacle_mask=["position", "radius"])
    # Definition of the obstacle.
    static_obst_dict = {
            "type": "sphere",
            "geometry": {"position": [2.0, 0.0, 0.0], "radius": 1.0},
    }
    obst1 = SphereObstacle(name="staticObst1", content_dict=static_obst_dict)
    # Definition of the goal.
    goal_dict = {
            "subgoal0": {
                "weight": 0.5,
                "is_primary_goal": True,
                "indices": [0, 1],
                "parent_link" : 0,
                "child_link" : 1,
                "desired_position": [3.5, 0.5],
                "epsilon" : 0.1,
                "type": "staticSubGoal"
            }
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    env.reset(pos=pos0, vel=vel0)
    env.add_sensor(full_sensor, [0])
    env.add_goal(goal.sub_goals()[0])
    env.add_obstacle(obst1)
    return (env, goal)


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
    robot_type = "pointRobot"
    # Optional reconfiguration of the planner with collision_geometry/finsler, remove for defaults.
    collision_geometry = "-2.0 / (x ** 1) * xdot ** 2"
    collision_finsler = "1.0/(x**2) * (1 - ca.heaviside(xdot))* xdot**2"
    planner = ParameterizedFabricPlanner(
            degrees_of_freedom,
            robot_type,
            collision_geometry=collision_geometry,
            collision_finsler=collision_finsler
    )
    collision_links = [1]
    self_collision_links = {}
    # The planner hides all the logic behind the function set_components.
    # Workaround set-components
    planner.set_components(
        collision_links,
        self_collision_links,
        goal,
        number_obstacles=0,
    )
    planner.concretize()
    return planner


def run_point_robot_urdf(n_steps=10000, render=True):
    """
    Set the gym environment, the planner and run point robot example.
    The initial zero action step is needed to initialize the sensor in the
    urdf environment.

    Params
    ----------
    n_steps
        Total number of simulation steps.
    render
        Boolean toggle to set rendering on (True) or off (False).
    """
    (env, goal) = initalize_environment(render)
    planner = set_planner(goal)

    action = np.array([0.0, 0.0, 0.0])
    ob, *_ = env.step(action)
    p.resetDebugVisualizerCamera(2, 0, 270.1, [0, 0, 3])

    for _ in range(n_steps):
        # Calculate action with the fabric planner, slice the states to drop Z-axis [3] information.
        ob_robot = ob['robot_0']

        width_res = 100
        height_res = 100
        img = p.getCameraImage(width_res, height_res, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        proj_rgb = np.reshape(img[2], (height_res, width_res, 4)) * 1. / 255.
        proj_depth = img[3]

        action[0:2] = planner.compute_action(
            q=ob_robot["joint_state"]["position"][0:2],
            qdot=ob_robot["joint_state"]["velocity"][0:2],
            x_goal_0=ob_robot['FullSensor']['goals'][0][0][0:2],
            weight_goal_0=goal.sub_goals()[0].weight(),
            #x_obst_0=ob_robot['FullSensor']['obstacles'][0][0][0:2],
            #radius_obst_0=ob_robot['FullSensor']['obstacles'][0][1],
            #edf_eval=edf(ob_robot["joint_state"]["position"][0:2], proj_rgb),
            J_edf_eval=edf_jacobian(ob_robot["joint_state"]["position"][0:2]),
            radius_body_1=np.array([0.2])
        )
        ob, *_, = env.step(action)
        # env.render(mode='human')
        plt.subplot(1, 2, 1)
        plt.imshow(proj_rgb)
        plt.subplot(1, 2, 2)
        plt.imshow(proj_depth)
        plt.show()
        pos_current = ob["robot_0"]["joint_state"]["position"]
        edf(pos_current, proj_rgb)
        kkk=1
    return {}

if __name__ == "__main__":
    res = run_point_robot_urdf(n_steps=1, render=True)
