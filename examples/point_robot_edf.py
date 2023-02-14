import gym
import numpy as np
import casadi as ca
import logging
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.goals.goal_composition import GoalComposition
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner
from fabrics.components.energies.execution_energies import ExecutionLagrangian
from fabrics.diffGeometry.diffMap import ExplicitDifferentialMap
from fabrics.components.leaves.geometry import GenericGeometryLeaf
from fabrics.helpers.variables import Variables
from fabrics.diffGeometry.energized_geometry import WeightedGeometry
import pybullet as p
import matplotlib.pyplot as plt
from scipy import ndimage

# Fabrics example for a 3D point mass robot. The fabrics planner uses a 2D point
# mass to compute actions for a simulated 3D point mass.
#
# todo: tune behavior.

logging.basicConfig(level=logging.ERROR)


class EDFGeometryLeaf(GenericGeometryLeaf):
    def __init__(
        self,
        parent_variables: Variables,
    ):
        phi = ca.SX.sym("edf_phi", 1)
        super().__init__(
            parent_variables,
            f"edf_leaf",
            phi,
        )
        self.set_forward_map()

    def set_forward_map(self):
        J = ca.transpose(ca.SX.sym("J_edf", 2))
        Jdot = ca.transpose(ca.SX.sym("Jdot_edf", 2))
        explicit_jacobians = {
            "phi_edf": self._forward_kinematics,
            "J_edf": J,
            "Jdot_edf": Jdot,
        }
        self._parent_variables.add_parameters(explicit_jacobians)
        self._forward_map =  ExplicitDifferentialMap(
            self._forward_kinematics,
            self._parent_variables,
            J=J,
            Jdot=Jdot,
        )

    def map(self):
        return self._forward_map

class EDFPlanner(ParameterizedFabricPlanner):
    def set_components(
        self,
        collision_links: list,
        goal: GoalComposition = None,
    ):
        geometry = EDFGeometryLeaf(self._variables)
        geometry.set_geometry(self.config.collision_geometry)
        geometry.set_finsler_structure(self.config.collision_finsler)
        self.add_leaf(geometry)
        if goal:
            self.set_goal_component(goal)
            # Adds default execution energy
            execution_energy = ExecutionLagrangian(self._variables)
            self.set_execution_energy(execution_energy)
            # Sets speed control
            self.set_speed_control()
        return geometry

def edf(pos, proj_rgb) -> tuple:
    #to binary image, obstacle are red
    proj_r = proj_rgb[:, :, 1]
    proj_bin = ((1-proj_r) > 0.9)

    if logging.root.level <= 10:
        plt.subplot(1, 3, 1)
        plt.imshow(proj_r)

        plt.subplot(1, 3, 2)
        plt.imshow(proj_bin)

    pixel_to_meter_ratio = 0.1
    offset = np.array([6.5, 5])
    direction_multiplier = np.array([1, -1])
    
    dist_map = ndimage.distance_transform_edt(1-proj_bin)
    dist_map = dist_map * pixel_to_meter_ratio # /100 pixels * 20 in meters

    # convert pos to pixels
    pos_p = np.rint((direction_multiplier * pos + offset)/pixel_to_meter_ratio)
    pos_p = [int(pos_pi) for pos_pi in pos_p]

    # dist_map = dist_map_t
    if logging.root.level <= 10:
        plt.subplot(1, 3, 3)
        plt.imshow(dist_map)
        plt.show()

    # index in map
    dist_pos = dist_map[int(pos_p[1]), int(pos_p[0])]

    gradient_map = np.gradient(dist_map)
    gradient_x = gradient_map[1]/pixel_to_meter_ratio
    gradient_y = -gradient_map[0]/pixel_to_meter_ratio
    grad_x_pos = gradient_x[pos_p[1], pos_p[0]]
    grad_y_pos = gradient_y[pos_p[1], pos_p[0]]
    return (dist_pos, grad_x_pos, grad_y_pos) #dist_map

def get_top_view_image(save=False):
    try:
        proj_rgb = np.load("proj_rgb.npy")
        proj_depth = np.load("proj_depth.npy")
    except FileNotFoundError:
        width_res = 130
        height_res = 100
        img = p.getCameraImage(width_res, height_res, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        proj_rgb = np.reshape(img[2], (height_res, width_res, 4)) * 1. / 255.
        proj_depth = img[3]
        if save:
            np.save('proj_rgb', proj_rgb)
            np.save('proj_depth', proj_depth)
    return proj_rgb, proj_depth

def test_edf_evaluation(proj_rgb):
    edf_test, *_ = edf([2.0, 0.0], proj_rgb)
    assert edf_test == 0.0
    _, edf_gradient_x, edf_gradient_y = edf(
        [4.0, 2.0],
        proj_rgb
    )
    assert edf_gradient_x >= 0
    assert edf_gradient_y >= 0
    _, edf_gradient_x, edf_gradient_y = edf(
        [4.0, -2.0],
        proj_rgb
    )
    assert edf_gradient_x >= 0
    assert edf_gradient_y <= 0
    _, edf_gradient_x, edf_gradient_y = edf(
        [-4.0, -2.0],
        proj_rgb
    )
    assert edf_gradient_x <= 0
    assert edf_gradient_y <= 0
    _, edf_gradient_x, edf_gradient_y = edf(
        [-4.0, 2.0],
        proj_rgb
    )
    assert edf_gradient_x <= 0
    assert edf_gradient_y >= 0

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
    vel0 = np.array([0.0, 0.0, 0.0])
    full_sensor = FullSensor(goal_mask=["position"], obstacle_mask=["position", "radius"])
    # Definition of the obstacle.
    static_obst_dict_1 = {
            "type": "sphere",
            "geometry": {"position": [2.0, 1.2, 0.0], "radius": 1.0},
    }
    static_obst_dict_2 = {
            "type": "sphere",
            "geometry": {"position": [0.5, -0.8, 0.0], "radius": 0.4},
    }
    static_obst_dict_3 = {
            "type": "sphere",
            "geometry": {"position": [0.0, 4.0, 0.0], "radius": 1.5},
    }
    obst1 = SphereObstacle(name="staticObst1", content_dict=static_obst_dict_1)
    obst2 = SphereObstacle(name="staticObst2", content_dict=static_obst_dict_2)
    obst3 = SphereObstacle(name="staticObst3", content_dict=static_obst_dict_3)
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
    env.add_obstacle(obst2)
    env.add_obstacle(obst3)
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
    collision_geometry = "-0.8 / (x ** 2) * xdot ** 2"
    collision_finsler = "0.5/(x ** 2) * (1 - ca.heaviside(xdot))* xdot**2"
    planner = EDFPlanner(
            degrees_of_freedom,
            robot_type,
            collision_geometry=collision_geometry,
            collision_finsler=collision_finsler
    )
    collision_links = [1]
    geometry = planner.set_components(
        collision_links,
        goal,
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
    p.resetDebugVisualizerCamera(5, 0, 270.1, [0, 0, 0])
    input("Make sure that the pybullet window is in default window size. Then press any key.")
    planner = set_planner(goal)

    action = np.array([0.0, 0.0, 0.0])
    ob, *_ = env.step(action)
    proj_rgb, proj_depth = get_top_view_image(save=True)

    if logging.root.level <= 10:
        plt.subplot(1, 2, 1)
        plt.imshow(proj_rgb)
        plt.subplot(1, 2, 2)
        plt.imshow(proj_depth)
        plt.show()
        test_edf_evaluation(proj_rgb)

    for _ in range(n_steps):
        # Calculate action with the fabric planner, slice the states to drop Z-axis [3] information.
        ob_robot = ob['robot_0']
        proj_rgb, proj_depth = get_top_view_image(save=False)

        edf_phi, edf_gradient_x, edf_gradient_y = edf(
            ob_robot['joint_state']['position'][0:2],
            proj_rgb
        )

        
        edf_gradient = np.array([edf_gradient_x, edf_gradient_y])        
        #print(edf_gradient)
        action[0:2] = planner.compute_action(
            q=ob_robot["joint_state"]["position"][0:2],
            qdot=ob_robot["joint_state"]["velocity"][0:2],
            x_goal_0=ob_robot['FullSensor']['goals'][0][0][0:2],
            weight_goal_0=goal.sub_goals()[0].weight(),
            #x_obst_0=ob_robot['FullSensor']['obstacles'][0][0][0:2],
            #radius_obst_0=ob_robot['FullSensor']['obstacles'][0][1],
            phi_edf=edf_phi-0.2,
            J_edf=edf_gradient,
            Jdot_edf=np.zeros(2),
            radius_body_1=np.array([0.2])
        )
        ob, *_, = env.step(action)
    return {}

if __name__ == "__main__":
    res = run_point_robot_urdf(n_steps=100000, render=True)
