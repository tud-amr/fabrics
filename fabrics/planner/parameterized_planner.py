import logging
from copy import deepcopy
from typing import Dict, List, Optional
import os

import casadi as ca
import deprecation
import numpy as np

from forwardkinematics.fksCommon.fk import ForwardKinematics
from forwardkinematics.urdfFks.urdfFk import LinkNotInURDFError
from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.goals.sub_goal import SubGoal
from pyquaternion import Quaternion

from fabrics import __version__
from fabrics.components.energies.execution_energies import ExecutionLagrangian
from fabrics.components.leaves.attractor import GenericAttractor
from fabrics.components.leaves.dynamic_attractor import GenericDynamicAttractor
from fabrics.components.leaves.dynamic_geometry import (
    DynamicObstacleLeaf, GenericDynamicGeometryLeaf)
from fabrics.components.leaves.geometry import (AvoidanceLeaf,
                                                CapsuleCuboidLeaf,
                                                CapsuleSphereLeaf,
                                                ESDFGeometryLeaf,
                                                GenericGeometryLeaf, LimitLeaf,
                                                ObstacleLeaf,
                                                PlaneConstraintGeometryLeaf,
                                                SelfCollisionLeaf,
                                                SphereCuboidLeaf)
from fabrics.components.leaves.leaf import Leaf
from fabrics.diffGeometry.diffMap import (DifferentialMap,
                                          DynamicDifferentialMap)
from fabrics.diffGeometry.energized_geometry import WeightedGeometry
from fabrics.diffGeometry.energy import Lagrangian
from fabrics.diffGeometry.geometry import Geometry
from fabrics.diffGeometry.speedControl import Damper
from fabrics.helpers.casadiFunctionWrapper import CasadiFunctionWrapper
from fabrics.helpers.constants import eps
from fabrics.helpers.exceptions import ExpressionSparseError
from fabrics.helpers.functions import is_sparse, parse_symbolic_input
from fabrics.helpers.geometric_primitives import Sphere
from fabrics.helpers.variables import Variables
from fabrics.planner.configuration_classes import (FabricPlannerConfig,
                                                   ProblemConfiguration)


class InvalidRotationAnglesError(Exception):
    pass

class LeafNotFoundError(Exception):
    pass


@deprecation.deprecated(deprecated_in="0.8.8", removed_in="0.9",
                        current_version=__version__,
                        details="Remove the goal attribute angle and rotate the position before passing it into compute_action.")
def compute_rotation_matrix(angles) -> np.ndarray:
    if isinstance(angles, float):
        angle = angles
        return np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    elif isinstance(angles, list) and len(angles) == 4:
        quaternion = Quaternion(angles)
        return quaternion.rotation_matrix
    elif isinstance(angles, ca.SX):
        return angles
    else:
        raise(InvalidRotationAnglesError)


class ParameterizedFabricPlanner(object):
    
    _dof: int
    _config: FabricPlannerConfig
    _problem_configuration : ProblemConfiguration
    leaves: Dict[str, Leaf]
    _ref_sign: int


    def __init__(self, dof: int, forward_kinematics: ForwardKinematics, **kwargs):
        self._dof = dof
        self._config = FabricPlannerConfig(**kwargs)
        self._forward_kinematics = forward_kinematics
        self.initialize_joint_variables()
        self.set_base_geometry()
        self._target_velocity = np.zeros(self._geometry.x().size()[0])
        self._ref_sign = 1
        self.leaves = {}

    """ INITIALIZING """

    def load_fabrics_configuration(self, fabrics_configuration: dict):
        self._config = FabricPlannerConfig(**fabrics_configuration)

    def initialize_joint_variables(self):
        q = ca.SX.sym("q", self._dof)
        qdot = ca.SX.sym("qdot", self._dof)
        self._variables = Variables(state_variables={"q": q, "qdot": qdot})

    def set_base_geometry(self):
        q = self._variables.position_variable()
        qdot = self._variables.velocity_variable()
        new_parameters, base_energy =  parse_symbolic_input(self._config.base_energy, q, qdot)
        self._variables.add_parameters(new_parameters)
        base_geometry = Geometry(h=ca.SX(np.zeros(self._dof)), var=self.variables)
        base_lagrangian = Lagrangian(base_energy, var=self._variables)
        self._geometry = WeightedGeometry(g=base_geometry, le=base_lagrangian)

    @property
    def variables(self) -> Variables:
        return self._variables

    @property
    def config(self) -> FabricPlannerConfig:
        return self._config

    """ ADDING COMPONENTS """

    def add_geometry(
        self, forward_map: DifferentialMap, lagrangian: Lagrangian, geometry: Geometry
    ) -> None:
        assert isinstance(forward_map, DifferentialMap)
        assert isinstance(lagrangian, Lagrangian)
        assert isinstance(geometry, Geometry)
        weighted_geometry = WeightedGeometry(g=geometry, le=lagrangian)
        self.add_weighted_geometry(forward_map, weighted_geometry)

    def add_dynamic_geometry(
        self,
        forward_map: DifferentialMap,
        dynamic_map: DynamicDifferentialMap,
        geometry_map: DifferentialMap,
        lagrangian: Lagrangian,
        geometry: Geometry,
    ) -> None:
        assert isinstance(forward_map, DifferentialMap)
        assert isinstance(geometry_map, DifferentialMap)
        assert isinstance(dynamic_map, DynamicDifferentialMap)
        assert isinstance(lagrangian, Lagrangian)
        assert isinstance(geometry, Geometry)
        weighted_geometry = WeightedGeometry(g=geometry, le=lagrangian, ref_names=dynamic_map.ref_names())
        pwg1 = weighted_geometry.pull(geometry_map)
        pwg2 = pwg1.dynamic_pull(dynamic_map)
        pwg3 = pwg2.pull(forward_map)
        self._geometry += pwg3

    def add_weighted_geometry(
        self, forward_map: DifferentialMap, weighted_geometry: WeightedGeometry
    ) -> None:
        assert isinstance(forward_map, DifferentialMap)
        assert isinstance(weighted_geometry, WeightedGeometry)
        pulled_geometry = weighted_geometry.pull(forward_map)
        self._geometry += pulled_geometry
        #self._refTrajs = joinRefTrajs(self._refTrajs, eg._refTrajs)
        self._variables = self._variables + pulled_geometry._vars

    def add_leaf(self, leaf: Leaf, prime_leaf: bool= False) -> None:
        if isinstance(leaf, GenericAttractor):
            self.add_forcing_geometry(leaf.map(), leaf.lagrangian(), leaf.geometry(), prime_leaf)
        elif isinstance(leaf, GenericDynamicAttractor):
            self.add_dynamic_forcing_geometry(leaf.map(), leaf.dynamic_map(), leaf.lagrangian(), leaf.geometry(), leaf._xdot_ref, prime_leaf)
        elif isinstance(leaf, GenericGeometryLeaf):
            self.add_geometry(leaf.map(), leaf.lagrangian(), leaf.geometry())
        elif isinstance(leaf, GenericDynamicGeometryLeaf):
            self.add_dynamic_geometry(leaf.map(), leaf.dynamic_map(), leaf.geometry_map(), leaf.lagrangian(), leaf.geometry())
        self.leaves[leaf._leaf_name] = leaf

    def get_leaves(self, leaf_names:list) -> List[Leaf]:
        leaves = []
        for leaf_name in leaf_names:
            if leaf_name not in self.leaves:
                error_message = f"Leaf with name {leaf_name} not in leaves.\n"
                error_message = f"Possible leaves are {list(self.leaves.keys())}."
                raise LeafNotFoundError(error_message)
            leaves.append(self.leaves[leaf_name])
        return leaves

    def add_forcing_geometry(
        self,
        forward_map: DifferentialMap,
        lagrangian: Lagrangian,
        geometry: Geometry,
        prime_forcing_leaf: bool,
    ) -> None:
        assert isinstance(forward_map, DifferentialMap)
        assert isinstance(lagrangian, Lagrangian)
        assert isinstance(geometry, Geometry)
        if not hasattr(self, '_forced_geometry'):
            self._forced_geometry = deepcopy(self._geometry)
        self._forcing_term = geometry.pull(forward_map)
        self._attractor_geometry = WeightedGeometry(
            g=geometry, le=lagrangian
        ).pull(forward_map)
        self._attractor_geometry.concretize()
        self._forced_geometry += WeightedGeometry(
            g=geometry, le=lagrangian
        ).pull(forward_map)

        if prime_forcing_leaf:
            self._forced_variables = geometry._vars
            self._forced_forward_map = forward_map
        self._variables = self._variables + self._forced_geometry._vars
        self._geometry.concretize()
        self._forcing_term.concretize()
        self._forced_geometry.concretize(ref_sign=self._ref_sign)

    def add_dynamic_forcing_geometry(
        self,
        forward_map: DifferentialMap,
        dynamic_map: DifferentialMap,
        lagrangian: Lagrangian,
        geometry: Geometry,
        target_velocity: ca.SX,
        prime_forcing_leaf: bool,
    ) -> None:
        assert isinstance(forward_map, DifferentialMap)
        assert isinstance(dynamic_map, DynamicDifferentialMap)
        assert isinstance(lagrangian, Lagrangian)
        assert isinstance(geometry, Geometry)
        assert isinstance(target_velocity, ca.SX)
        if not hasattr(self, '_forced_geometry'):
            self._forced_geometry = deepcopy(self._geometry)
        wg = WeightedGeometry(g=geometry, le=lagrangian)
        pwg = wg.dynamic_pull(dynamic_map)
        ppwg = pwg.pull(forward_map)
        self._forced_geometry += ppwg
        if prime_forcing_leaf:
            self._forced_variables = geometry._vars
            self._forced_forward_map = forward_map
        self._variables = self._variables + self._forced_geometry._vars
        self._target_velocity += ca.mtimes(ca.transpose(forward_map._J), target_velocity)
        self._ref_sign = -1
        self._geometry.concretize()
        self._forced_geometry.concretize(ref_sign=self._ref_sign)

    def set_execution_energy(self, execution_lagrangian: Lagrangian):
        print(f"Setting exection energy")
        assert isinstance(execution_lagrangian, Lagrangian)
        composed_geometry = Geometry(s=self._geometry)
        self._execution_lagrangian = execution_lagrangian
        self._execution_geometry = WeightedGeometry(
            g=composed_geometry, le=execution_lagrangian
        )
        self._execution_geometry.concretize()
        try:
            forced_geometry = Geometry(s=self._forced_geometry)
            self._forced_speed_controlled_geometry = WeightedGeometry(
                g=forced_geometry, le=execution_lagrangian
            )
            self._forced_speed_controlled_geometry.concretize()
        except AttributeError as exception:
            logging.warning(f"Error setting the execution energy {exception}")

    def set_speed_control(self):
        x_psi = self._forced_variables.position_variable()
        dm_psi = self._forced_forward_map
        exLag = self._execution_lagrangian
        a_ex = ca.SX.sym("a_ex", 1)
        a_le = ca.SX.sym("a_le", 1)
        beta_expression = self.config.damper_beta
        eta_expression = self.config.damper_eta
        self._damper = Damper(beta_expression, eta_expression, x_psi, dm_psi, exLag._l)
        self._variables.add_parameters(self._damper.symbolic_parameters())

    def get_forward_kinematics(self, link_name, position_only: bool = True) -> ca.SX:
        if isinstance(link_name, ca.SX):
            return link_name

        fk = self._forward_kinematics.casadi(
                self._variables.position_variable(),
                link_name,
                position_only=position_only
            )
        return fk

    def add_capsule_sphere_geometry(
            self,
            obstacle_name: str,
            capsule_name: str,
            tf_capsule_origin: ca.SX,
            capsule_length: float
            ) -> None:
        tf_origin_center_0 = np.identity(4)
        tf_origin_center_0[2][3] = capsule_length / 2
        tf_center_0 = ca.mtimes(tf_capsule_origin, tf_origin_center_0)
        tf_origin_center_1 = np.identity(4)
        tf_origin_center_1[2][3] = - capsule_length / 2
        tf_center_1 = ca.mtimes(tf_capsule_origin, tf_origin_center_1)
        capsule_sphere_leaf = CapsuleSphereLeaf(
            self._variables,
            capsule_name,
            obstacle_name,
            tf_center_0[0:3,3],
            tf_center_1[0:3,3],
        )
        capsule_sphere_leaf.set_geometry(self.config.collision_geometry)
        capsule_sphere_leaf.set_finsler_structure(self.config.collision_finsler)
        self.add_leaf(capsule_sphere_leaf)

    def add_capsule_cuboid_geometry(
            self,
            obstacle_name: str,
            capsule_name: str,
            tf_capsule_origin: ca.SX,
            capsule_length: float
    ):
        tf_origin_center_0 = np.identity(4)
        tf_origin_center_0[2][3] = capsule_length / 2
        tf_center_0 = ca.mtimes(tf_capsule_origin, tf_origin_center_0)
        tf_origin_center_1 = np.identity(4)
        tf_origin_center_1[2][3] = - capsule_length / 2
        tf_center_1 = ca.mtimes(tf_capsule_origin, tf_origin_center_1)
        capsule_cuboid_leaf = CapsuleCuboidLeaf(
            self._variables,
            capsule_name,
            obstacle_name,
            tf_center_0[0:3,3],
            tf_center_1[0:3,3],
        )
        capsule_cuboid_leaf.set_geometry(self.config.collision_geometry)
        capsule_cuboid_leaf.set_finsler_structure(self.config.collision_finsler)
        self.add_leaf(capsule_cuboid_leaf)

    def add_spherical_obstacle_geometry(
            self,
            obstacle_name: str,
            collision_link_name: str,
            forward_kinematics: ca.SX,
            ) -> None:
        """
        Add a spherical obstacle geometry to the fabrics planner.

        Parameters
        ----------
        obstacle_name : str
            The name of the obstacle to be added.
        collision_link_name : str
            The name of the robot's collision link that the obstacle is associated with.
        forward_kinematics : ca.SX
            The forward kinematics expression representing the obstacle's position.

        Returns
        -------
        None

        Notes
        -----
        - The `forward_kinematics` should be a symbolic
            expression using CasADi SX for the obstacle's position.
        - The `collision_geometry` and `collision_finsler` configurations
            must be set before adding the obstacle.
        - After adding the obstacle, it will be included in the robot's
            configuration and affect its motion planning.

        Example
        -------
        obstacle_name = "Sphere_Obstacle"
        collision_link_name = "Link1"
        forward_kinematics = ca.SX([fk_x, fk_y, fk_z])
        robot.add_spherical_obstacle_geometry(
            obstacle_name,
            collision_link_name,
            forward_kinematics
        )
        """
        geometry = ObstacleLeaf(
            self._variables,
            forward_kinematics,
            obstacle_name,
            collision_link_name,
        )
        geometry.set_geometry(self.config.collision_geometry)
        geometry.set_finsler_structure(self.config.collision_finsler)
        self.add_leaf(geometry)

    def add_dynamic_spherical_obstacle_geometry(
            self,
            obstacle_name: str,
            collision_link_name: str,
            forward_kinematics: ca.SX,
            reference_parameters: dict,
            dynamic_obstacle_dimension: int = 3,
            ) -> None:
        geometry = DynamicObstacleLeaf(
            self._variables,
            forward_kinematics[0:dynamic_obstacle_dimension],
            obstacle_name,
            collision_link_name,
            reference_parameters=reference_parameters
        )
        geometry.set_geometry(self.config.collision_geometry)
        geometry.set_finsler_structure(self.config.collision_finsler)
        self.add_leaf(geometry)

    def add_plane_constraint(
            self,
            constraint_name: str,
            collision_link_name: str,
            forward_kinematics: ca.SX,
            ) -> None:

        geometry = PlaneConstraintGeometryLeaf(
            self._variables,
            constraint_name,
            collision_link_name,
            forward_kinematics,
        )
        geometry.set_geometry(self.config.geometry_plane_constraint)
        geometry.set_finsler_structure(self.config.finsler_plane_constraint)
        self.add_leaf(geometry)

    def add_cuboid_obstacle_geometry(
            self,
            obstacle_name: str,
            collision_link_name: str,
            forward_kinematics: ca.SX,
            ) -> None:

        geometry = SphereCuboidLeaf(
            self._variables,
            forward_kinematics,
            obstacle_name,
            collision_link_name,
        )
        geometry.set_geometry(self.config.collision_geometry)
        geometry.set_finsler_structure(self.config.collision_finsler)
        self.add_leaf(geometry)
    
    def add_esdf_geometry(
            self,
            collision_link_name: str,
            ) -> None:
        fk = self.get_forward_kinematics(collision_link_name)
        geometry = ESDFGeometryLeaf(self._variables, collision_link_name, fk)
        geometry.set_geometry(self.config.collision_geometry)
        geometry.set_finsler_structure(self.config.collision_finsler)
        self.add_leaf(geometry)

    def add_spherical_self_collision_geometry(
            self,
            collision_link_1: str,
            collision_link_2: str,
            ) -> None:
        fk_1 = self.get_forward_kinematics(collision_link_1)
        fk_2 = self.get_forward_kinematics(collision_link_2)
        fk = fk_2 - fk_1
        if is_sparse(fk):
            message = (
                    f"Expression {fk} for links {collision_link_1} "
                    "and {collision_link_2} is sparse and thus skipped."
            )
            logging.warning(message.format_map(locals()))
        geometry = SelfCollisionLeaf(self._variables, fk, collision_link_1, collision_link_2)
        geometry.set_geometry(self.config.self_collision_geometry)
        geometry.set_finsler_structure(self.config.self_collision_finsler)
        self.add_leaf(geometry)

    def add_limit_geometry(
            self,
            joint_index: int,
            limits: list,
            ) -> None:
        lower_limit_geometry = LimitLeaf(self._variables, joint_index, limits[0], 0)
        lower_limit_geometry.set_geometry(self.config.limit_geometry)
        lower_limit_geometry.set_finsler_structure(self.config.limit_finsler)
        upper_limit_geometry = LimitLeaf(self._variables, joint_index, limits[1], 1)
        upper_limit_geometry.set_geometry(self.config.limit_geometry)
        upper_limit_geometry.set_finsler_structure(self.config.limit_finsler)
        self.add_leaf(lower_limit_geometry)
        self.add_leaf(upper_limit_geometry)

    def load_problem_configuration(self, problem_configuration: ProblemConfiguration):
        self._problem_configuration = ProblemConfiguration(**problem_configuration)
        for obstacle in self._problem_configuration.environment.obstacles:
            self._variables.add_parameters(obstacle.sym_parameters)

        self.set_collision_avoidance()
        #self.set_self_collision_avoidance()
        self.set_joint_limits()
        if self._config.forcing_type in ['forced', 'speed-controlled', 'forced-energized']:
            self.set_goal_component(self._problem_configuration.goal_composition)
        if self._config.forcing_type in ['speed-controlled', 'execution-energy', 'forced-energized']:
            execution_energy = ExecutionLagrangian(self._variables)
            self.set_execution_energy(execution_energy)
        if self._config.forcing_type in ['speed-controlled']:
            self.set_speed_control()

    def set_joint_limits(self):
        limits = np.zeros((self._dof, 2))
        limits[:, 0] = self._problem_configuration.joint_limits.lower_limits
        limits[:, 1] = self._problem_configuration.joint_limits.upper_limits
        limits = limits.tolist()
        for joint_index in range(len(limits)):
            self.add_limit_geometry(joint_index, limits[joint_index])

    def set_self_collision_avoidance(self) -> None:
        if not self._problem_configuration.robot_representation.self_collision_pairs:
            return
        for link_name,  paired_links_names in self._problem_configuration.robot_representation.self_collision_pairs.items():
            link = self._problem_configuration.robot_representation.collision_links[link_name]
            for paired_link_name in paired_links_names:
                paired_link = self._problem_configuration.robot_representation.collision_links[paired_link_name]
                if isinstance(link, Sphere) and isinstance(paired_link, Sphere):
                    self.add_spherical_self_collision_geometry(
                            paired_link_name,
                            link_name,
                    )


    def set_collision_avoidance(self) -> None:
        if not self._problem_configuration.robot_representation.collision_links:
            return
        for link_name, collision_link in self._problem_configuration.robot_representation.collision_links.items():
            fk = self.get_forward_kinematics(link_name, position_only=False)
            if fk.shape == (3, 3):
                fk_augmented = ca.SX.eye(4)
                fk_augmented[0:2, 0:2] = fk[0:2, 0:2]
                fk_augmented[0:2, 3] = fk[0:2, 2]
                fk = fk_augmented
            if fk.shape == (4, 4) and is_sparse(fk[0:3, 3]):
                message = (
                        f"Expression {fk[0:3, 3]} for link {link_name} "
                        "is sparse and thus skipped."
                )
                logging.warning(message.format_map(locals()))
                continue
            elif fk.shape != (4, 4) and is_sparse(fk):
                message = (
                        f"Expression {fk} for link {link_name} "
                        "is sparse and thus skipped."
                )
                logging.warning(message.format_map(locals()))
                continue
            collision_link.set_origin(fk)
            self._variables.add_parameters(collision_link.sym_parameters)
            self._variables.add_parameters_values(collision_link.parameters)
            for obstacle in self._problem_configuration.environment.obstacles:
                distance = collision_link.distance(obstacle)
                leaf_name = f"{link_name}_{obstacle.name}_leaf"
                leaf = AvoidanceLeaf(self._variables, leaf_name, distance)
                leaf.set_geometry(self.config.collision_geometry)
                leaf.set_finsler_structure(self.config.collision_finsler)
                self.add_leaf(leaf)
            """
            for i in range(self._problem_configuration.environment.number_spheres['dynamic']):
                obstacle_name = f"obst_dynamic_{i}"
                if isinstance(collision_link, Sphere):
                    self.add_dynamic_spherical_obstacle_geometry(
                            obstacle_name,
                            link_name,
                            fk,
                            reference_parameter_list[i],
                            dynamic_obstacle_dimension=dynamic_obstacle_dimension,
                    )
            for i in range(self._problem_configuration.environment.number_planes):
                constraint_name = f"constraint_{i}"
                if isinstance(collision_link, Sphere):
                    self.add_plane_constraint(constraint_name, link_name, fk)

            for i in range(self._problem_configuration.environment.number_cuboids['static']):
                obstacle_name = f"obst_cuboid_{i}"
                if isinstance(collision_link, Sphere):
                    self.add_cuboid_obstacle_geometry(obstacle_name, link_name, fk)
            """




    def set_components(
        self,
        collision_links: Optional[list] = None,
        self_collision_pairs: Optional[dict] = None,
        collision_links_esdf: Optional[list] = None,
        goal: Optional[GoalComposition] = None,
        limits: Optional[list] = None,
        number_obstacles: int = 1,
        number_dynamic_obstacles: int = 0,
        number_obstacles_cuboid: int = 0,
        number_plane_constraints: int = 0,
        dynamic_obstacle_dimension: int = 3,
    ):
        collision_links = collision_links or []
        collision_links_esdf = collision_links_esdf or []
        self_collision_pairs = self_collision_pairs or {}

        reference_parameter_list = []
        for i in range(number_dynamic_obstacles):
            reference_parameters = {
                f"x_obst_dynamic_{i}": ca.SX.sym(f"x_obst_dynamic_{i}", dynamic_obstacle_dimension),
                f"xdot_obst_dynamic_{i}": ca.SX.sym(f"xdot_obst_dynamic_{i}", dynamic_obstacle_dimension),
                f"xddot_obst_dynamic_{i}": ca.SX.sym(f"xddot_obst_dynamic_{i}", dynamic_obstacle_dimension),
            }
            reference_parameter_list.append(reference_parameters)
        for collision_link in collision_links:
            fk = self.get_forward_kinematics(collision_link)
            if is_sparse(fk):
                message = (
                        f"Expression {fk} for link {collision_link} "
                        "is sparse and thus skipped."
                )
                logging.warning(message.format_map(locals()))
                continue
            for i in range(number_obstacles):
                obstacle_name = f"obst_{i}"
                self.add_spherical_obstacle_geometry(obstacle_name, collision_link, fk)
            for i in range(number_dynamic_obstacles):
                obstacle_name = f"obst_dynamic_{i}"
                self.add_dynamic_spherical_obstacle_geometry(
                        obstacle_name,
                        collision_link,
                        fk,
                        reference_parameter_list[i],
                        dynamic_obstacle_dimension=dynamic_obstacle_dimension,
                )
            for i in range(number_plane_constraints):
                constraint_name = f"constraint_{i}"
                self.add_plane_constraint(constraint_name, collision_link, fk)

            for i in range(number_obstacles_cuboid):
                obstacle_name = f"obst_cuboid_{i}"
                self.add_cuboid_obstacle_geometry(obstacle_name, collision_link, fk)


        for collision_link in collision_links_esdf:
            self.add_esdf_geometry(collision_link)

        for self_collision_key, self_collision_list in self_collision_pairs.items():
            for self_collision_link in self_collision_list:
                self.add_spherical_self_collision_geometry(
                        self_collision_link,
                        self_collision_key,
                )

        if limits:
            for joint_index in range(len(limits)):
                self.add_limit_geometry(joint_index, limits[joint_index])

        if goal:
            self.set_goal_component(goal)
            # Adds default execution energy
            execution_energy = ExecutionLagrangian(self._variables)
            self.set_execution_energy(execution_energy)
            # Sets speed control
            self.set_speed_control()
        else:
            execution_energy = ExecutionLagrangian(self._variables)
            self.set_execution_energy(execution_energy)

    def get_differential_map(self, sub_goal_index: int, sub_goal: SubGoal):
        if sub_goal.type() == 'staticJointSpaceSubGoal':
            return self._variables.position_variable()[sub_goal.indices()]
        else:
            fk_child = self.get_forward_kinematics(sub_goal.child_link())
            try:
                fk_parent = self.get_forward_kinematics(sub_goal.parent_link())
            except LinkNotInURDFError as e:
                fk_parent = ca.SX(np.zeros(3))
            angles = sub_goal.angle()

            if angles and isinstance(angles, list) and len(angles) == 4:
                logging.warning(
                    "Subgoal attribute 'angle' deprecated. " \
                    +"Remove the goal attribute angle and rotate the" \
                    +"position before passing it into"\
                    +"compute_action."
                )
                angles = ca.SX.sym(f"angle_goal_{sub_goal_index}", 3, 3)
                self._variables.add_parameter(f'angle_goal_{sub_goal_index}', angles)
                # rotation
                R = compute_rotation_matrix(angles)
                fk_child = ca.mtimes(R, fk_child)
                fk_parent = ca.mtimes(R, fk_parent)
            elif angles:
                logging.warning(
                    "Subgoal attribute 'angle' deprecated. " \
                    +"Remove the goal attribute angle and rotate the" \
                    +"position before passing it into"\
                    +"compute_action."
                )
                R = compute_rotation_matrix(angles)
                fk_child = ca.mtimes(R, fk_child)
                fk_parent = ca.mtimes(R, fk_parent)
            return fk_child[sub_goal.indices()] - fk_parent[sub_goal.indices()]



    def set_goal_component(self, goal: GoalComposition):
        # Adds default attractor
        for j, sub_goal in enumerate(goal.sub_goals()):
            fk_sub_goal = self.get_differential_map(j, sub_goal)
            if is_sparse(fk_sub_goal):
                raise ExpressionSparseError()
            if sub_goal.type() in ["analyticSubGoal", "splineSubGoal"]:
                attractor = GenericDynamicAttractor(self._variables, fk_sub_goal, f"goal_{j}")
            else:
                self._variables.add_parameter(f'x_goal_{j}', ca.SX.sym(f'x_goal_{j}', sub_goal.dimension()))
                attractor = GenericAttractor(self._variables, fk_sub_goal, f"goal_{j}")
            attractor.set_potential(self.config.attractor_potential)
            attractor.set_metric(self.config.attractor_metric)
            self.add_leaf(attractor, prime_leaf=sub_goal.is_primary_goal())


    def concretize(self, mode='acc', time_step=None):
        self._mode = mode
        if mode == 'vel':
            if not time_step:
                raise Exception("No time step passed in velocity mode.")
        self._geometry.concretize()
        if self._config.forcing_type in ['speed-controlled']:
            eta = self._damper.substitute_eta()
            a_ex = (
                eta * self._execution_geometry._alpha
                + (1 - eta) * self._forced_speed_controlled_geometry._alpha
            )
            beta_subst = self._damper.substitute_beta(-a_ex, -self._geometry._alpha)
            xddot = self._forced_geometry._xddot - (a_ex + beta_subst) * (
                self._geometry.xdot()
                - ca.mtimes(self._forced_geometry.Minv(), self._target_velocity)
            )
            #xddot = self._forced_geometry._xddot
        elif self._config.forcing_type in ['simply_damped']:
            geometry = self._geometry._xddot - self._geometry._alpha * self._geometry._vars.velocity_variable()
            forcing = -ca.mtimes(
                        self._geometry.Minv(),
                        self._attractor_geometry.f()
                    )
            damping = -ca.mtimes(
                        self._geometry.Minv(),
                        self._damper._beta * self._geometry._vars.velocity_variable()
                    )
            xddot = geometry + forcing + damping
        elif self._config.forcing_type == 'execution-energy':
            logging.warn("No forcing term, using pure geoemtry with energization.")
            #xddot = self._geometry._xddot - self._geometry._alpha * self._geometry._vars.velocity_variable()
            xddot = self._execution_geometry._xddot - self._execution_geometry._alpha * self._geometry._vars.velocity_variable()
        elif self._config.forcing_type == 'forced-energized':
            logging.warn("Using forced geoemtry with constant execution energy.")
            xddot = self._forced_speed_controlled_geometry._xddot - self._forced_speed_controlled_geometry._alpha * self._geometry._vars.velocity_variable()
        elif self._config.forcing_type == 'forced':
            logging.warn("No execution energy, using forced geomtry without speed regulation.")
            xddot = self._forced_geometry._xddot - self._geometry._alpha * self._geometry._vars.velocity_variable()
        elif self._config.forcing_type == 'pure-geometry':
            xddot = self._geometry._xddot
        else:
            raise Exception(f"Unknown forcing type {self._config.forcing_type}.")

        if mode == 'acc':
            self._funs = CasadiFunctionWrapper(
                "funs", self.variables, {"action" : xddot}
            )
        elif mode == 'vel':
            action = self._geometry.xdot() + time_step * xddot
            self._funs = CasadiFunctionWrapper(
                "funs", self.variables, {"action": action}
            )
            

    def serialize(self, file_name: str):
        """
        Serializes the fabric planner.

        The file can be loaded using the serialized_planner.
        Essentially, only the casadiFunctionWrapper is serialized using
        pickle.
        """
        self._funs.serialize(file_name)

    def export_as_xml(self, file_name: str):
        """
        Exports the pure casadi function in xml format.

        The generated file can be loaded in python, cpp or Matlab.
        You can use that using the syntax ca.Function.load(file_name).
        Note that passing arguments as dictionary is not supported then.
        """
        function = self._funs.function()
        function.save(file_name)


    def export_as_c(self, file_name: str):
        """
        Export the planner as c source code.
        """
        filepath, filename = os.path.split(file_name)
        if not filepath.endswith(os.sep):
            filepath += os.sep
        
        gen = ca.CodeGenerator(filename)
        gen.add(self._funs.function())
        gen.generate(filepath)

 
    """ RUNTIME METHODS """

    def compute_action(self, **kwargs):
        """
        Computes action based on the states passed.

        The variables passed are the joint states, and the goal position.
        The action is nullified if its magnitude is very large or very small.
        """
        evaluations = self._funs.evaluate(**kwargs)
        action = evaluations["action"]
        # Debugging
        #logging.debug(f"alhpa_forced_geometry: {evaluations['alpha_forced_geometry']}")
        #logging.debug(f"alpha_geometry: {evaluations['alpha_geometry']}")
        #logging.debug(f"beta : {evaluations['beta']}")
        action_magnitude = np.linalg.norm(action)
        if action_magnitude < eps:
            logging.warning(f"Fabrics: Avoiding small action with magnitude {action_magnitude}")
            action *= 0.0
        elif action_magnitude > 1/eps:
            logging.warning(f"Fabrics: Avoiding large action with magnitude {action_magnitude}")
            action *= 0.0
        return action


