import os
import time
import gym
import casadi as ca
import numpy as np
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from urdfenvs.robots.generic_urdf import GenericUrdfReacher
from urdfenvs.sensors.full_sensor import FullSensor
from urdfenvs.sensors.free_space_decomposition import FreeSpaceDecompositionSensor
from mpscenes.obstacles.sphere_obstacle import SphereObstacle
from mpscenes.obstacles.box_obstacle import BoxObstacle
from mpscenes.goals.goal_composition import GoalComposition
#from examples.point_robot_sensors import get_goal_sensors, get_obstacles_sensors
from fabrics.components.energies.execution_energies import ExecutionLagrangian
from fabrics.components.leaves.geometry import GenericGeometryLeaf
from fabrics.components.maps.parameterized_maps import ParameterizedGeometryMap
from fabrics.helpers.variables import Variables
from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

# Fabrics example for a 3D point mass robot. The fabrics planner uses a 2D point
# mass to compute actions for a simulated 3D point mass.
#
# todo: tune behavior.

NUMBER_OF_RAYS = 100

class ParameterizedFSDMap(ParameterizedGeometryMap):
    def __init__(
        self,
        var: Variables,
        fk,
        constraint_variable,
        radius_body_variable,
    ):
        phi = ca.fabs(ca.dot(constraint_variable[0:3], fk) + constraint_variable[3]) / ca.norm_2(constraint_variable[0:3]) - radius_body_variable

        #phi = ca.fabs(a*x + b*y + c*z + d) / ((a**2 + b**2 + c**2)**0.5)
        #phi = (
        #    ca.norm_2(fk - reference_variable)
        #    / (radius_variable + radius_body_variable)
        #    - 1
        #)
        super().__init__(phi, var)
class FreeSpaceDecompGeometryLeaf(GenericGeometryLeaf):
    def __init__(
            self,
            parent_variables: Variables,
            constraint_name: str,
            collision_link: str,
            collision_fk: ca.SX,
    ):
        self._collision_link = collision_link
        self._collision_fk = collision_fk
        self._constraint_name = constraint_name
        super().__init__(
            parent_variables,
            f"{collision_link}_{constraint_name}",
            collision_fk,
        )
        self.set_forward_map()

    def set_forward_map(self):
        q = self._parent_variables.position_variable()
        radius_body_name = f"radius_body_{self._collision_link}"
        if radius_body_name in self._parent_variables.parameters():
            radius_body_variable = self._parent_variables.parameters()[
                radius_body_name
            ]
        else:
            radius_body_variable = ca.SX.sym(radius_body_name, 1)
        if self._constraint_name in self._parent_variables.parameters():
            constraint_variable = self._parent_variables.parameters()[
                self._constraint_name
            ]
        else:
            constraint_variable = ca.SX.sym(self._constraint_name, 4)
        geo_parameters = {
            radius_body_name: radius_body_variable,
            self._constraint_name: constraint_variable,
        }
        self._parent_variables.add_parameters(geo_parameters)
        self._forward_map = ParameterizedFSDMap(
            self._parent_variables,
            self._forward_kinematics,
            constraint_variable,
            radius_body_variable
        )

    def map(self):
        return self._forward_map

class FSDPlanner(ParameterizedFabricPlanner):
    def set_components(self, collision_links: list = None,
                       self_collision_pairs: dict = None,
                       collision_links_esdf: list = None,
                       goal: GoalComposition = None,
                       limits: list = None,
                       number_obstacles: int = 0,
                       number_constraints: int = 10,
                       number_dynamic_obstacles: int = 0):
        for collision_link in collision_links:
            fk = self.get_forward_kinematics(collision_link)
            for i in range(number_constraints):
                constraint_name = f"constraint_{i}"
                geometry = FreeSpaceDecompGeometryLeaf(self._variables, constraint_name, collision_link, fk)
                geometry.set_geometry(self.config.collision_geometry)
                geometry.set_finsler_structure(self.config.collision_finsler)
                self.add_leaf(geometry)
        execution_energy = ExecutionLagrangian(self._variables)
        self.set_execution_energy(execution_energy)
        if goal:
            self.set_goal_component(goal)
            # Adds default execution energy
            execution_energy = ExecutionLagrangian(self._variables)
            self.set_execution_energy(execution_energy)
            # Sets speed control
            self.set_speed_control()

def get_goal_fsd():
    goal_dict = {
            "subgoal0": {
                "weight": 2.5,
                "is_primary_goal": True,
                "indices": [0, 1],
                "parent_link" : 'world',
                "child_link" : 'base_link',
                "desired_position": [3.5, 0.5],
                "epsilon" : 0.1,
                "type": "staticSubGoal"
            }
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    return goal

def get_obstacles_fsd():
    static_obst_dict = {
            "type": "sphere",
            "geometry": {"position": [2.0, -0.5, 0.15], "radius": 0.8},
    }
    obst1 = SphereObstacle(name="staticObst1", content_dict=static_obst_dict)
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [0.0, 1.0, 0.15], "radius": 0.3},
    }
    obst2 = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
    static_obst_dict = {
        "type": "sphere",
        "geometry": {"position": [3.0, 1.2, 0.15], "radius": 0.3},
    }
    obst3 = SphereObstacle(name="staticObst", content_dict=static_obst_dict)
    static_obst_dict = {
        "type": "box",
        "geometry": {"position": [2.0, 1.7, 0.15], "width": 0.3, "length": 0.2, "height": 0.3},
    }
    obst4 = BoxObstacle(name="staticObst", content_dict=static_obst_dict)
    return [obst1, obst2, obst3, obst4]

def initalize_environment(render):
    """
    Initializes the simulation environment.

    Adds an obstacle and goal visualizaion to the environment and
    steps the simulation once.
   j
    Params
    ----------
    render
        Boolean toggle to set rendering on (True) or off (False).
    """
    robots = [
        GenericUrdfReacher(urdf="pointRobot.urdf", mode="vel"),
    ]
    env: UrdfEnv  = gym.make(
        "urdf-env-v0",
        dt=0.01, robots=robots, render=render
    )
    # Set the initial position and velocity of the point mass.
    pos0 = np.array([-2.0, 0.5, 0.0])
    vel0 = np.array([0.1, 0.0, 0.0])
    full_sensor = FullSensor(goal_mask=["position", "weight"], obstacle_mask=["position", "size"])
    fsd_sensor = FreeSpaceDecompositionSensor('lidar_sensor_joint', max_radius=5, plotting_interval=100, nb_rays=NUMBER_OF_RAYS)
    # Definition of the obstacle.
    obstacles = get_obstacles_fsd()
    # Definition of the goal.
    goal = get_goal_fsd()
    env.reset(pos=pos0, vel=vel0)
    env.add_sensor(full_sensor, [0])
    env.add_sensor(fsd_sensor, [0])
    env.add_goal(goal.sub_goals()[0])
    for obst in obstacles:
        env.add_obstacle(obst)
    env.set_spaces()
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
    degrees_of_freedom = 3
    robot_type = "xyz"
    # Optional reconfiguration of the planner with collision_geometry/finsler, remove for defaults.
    collision_geometry = "-2.0 / (x ** 1) * xdot ** 2"
    collision_finsler = "1.0/(x**2) * (1 - ca.heaviside(xdot))* xdot**2"
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    with open(absolute_path + "/point_robot.urdf", "r") as file:
        urdf = file.read()
    planner = FSDPlanner(
            degrees_of_freedom,
            robot_type,
            urdf=urdf,
            root_link='world',
            end_link='base_link',
            collision_geometry=collision_geometry,
            collision_finsler=collision_finsler
    )
    collision_links = ['base_link']
    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        collision_links=collision_links,
        goal=goal,
        number_obstacles=1,
    )
    planner.concretize(mode='vel', time_step=0.01)
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
    env.reconfigure_camera(5, 0, 270.1, [0, 0, 0])
    planner = set_planner(goal)

    action = np.array([0.0, 0.0, 0.0])
    ob, *_ = env.step(action)

    for _ in range(n_steps):
        t0 = time.perf_counter()
        # Calculate action with the fabric planner, slice the states to drop Z-axis [3] information.
        ob_robot = ob['robot_0']
        q = ob_robot['joint_state']['position']

        action = planner.compute_action(
            q=ob_robot["joint_state"]["position"],
            qdot=ob_robot["joint_state"]["velocity"],
            x_goal_0=ob_robot['FullSensor']['goals'][2]['position'][0:2],
            weight_goal_0=ob_robot['FullSensor']['goals'][2]['weight'],
            radius_body_base_link=np.array([0.35]),
            constraint_0=ob_robot['FreeSpaceDecompSensor']['constraint_0'],
            constraint_1=ob_robot['FreeSpaceDecompSensor']['constraint_1'],
            constraint_2=ob_robot['FreeSpaceDecompSensor']['constraint_2'],
            constraint_3=ob_robot['FreeSpaceDecompSensor']['constraint_3'],
            constraint_4=ob_robot['FreeSpaceDecompSensor']['constraint_4'],
            constraint_5=ob_robot['FreeSpaceDecompSensor']['constraint_5'],
            constraint_6=ob_robot['FreeSpaceDecompSensor']['constraint_6'],
            constraint_7=ob_robot['FreeSpaceDecompSensor']['constraint_7'],
            constraint_8=ob_robot['FreeSpaceDecompSensor']['constraint_8'],
            constraint_9=ob_robot['FreeSpaceDecompSensor']['constraint_9'],
        )
        ob, *_, = env.step(action)
        t1 = time.perf_counter()
        print(t1-t0)
    return {}

if __name__ == "__main__":
    res = run_point_robot_urdf(n_steps=10000, render=True)
