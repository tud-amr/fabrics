from dataclasses import dataclass, field
from typing import Dict
import logging
import casadi as ca
from forwardkinematics.urdfFks.urdfFk import LinkNotInURDFError
import numpy as np
from copy import deepcopy
from fabrics.helpers.exceptions import ExpressionSparseError
from typing import List

from fabrics.helpers.variables import Variables
from fabrics.helpers.constants import eps
from fabrics.helpers.functions import is_sparse, parse_symbolic_input

from fabrics.diffGeometry.diffMap import DifferentialMap, DynamicDifferentialMap
from fabrics.diffGeometry.energy import Lagrangian
from fabrics.diffGeometry.geometry import Geometry
from fabrics.diffGeometry.energized_geometry import WeightedGeometry
from fabrics.diffGeometry.speedControl import Damper

from fabrics.helpers.casadiFunctionWrapper import CasadiFunctionWrapper

from fabrics.components.energies.execution_energies import ExecutionLagrangian
from fabrics.components.leaves.leaf import Leaf
from fabrics.components.leaves.attractor import GenericAttractor
from fabrics.components.leaves.dynamic_attractor import GenericDynamicAttractor
from fabrics.components.leaves.dynamic_geometry import DynamicObstacleLeaf, GenericDynamicGeometryLeaf
from fabrics.components.leaves.geometry import ObstacleLeaf, LimitLeaf, SelfCollisionLeaf, GenericGeometryLeaf, ESDFGeometryLeaf

from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.goals.sub_goal import SubGoal

from forwardkinematics.fksCommon.fk_creator import FkCreator
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

from pyquaternion import Quaternion

class InvalidRotationAnglesError(Exception):
    pass

class LeafNotFoundError(Exception):
    pass

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



@dataclass
class FabricPlannerConfig:
    base_energy: str = (
        "0.5 * 0.2 * ca.dot(xdot, xdot)"
    )
    collision_geometry: str = (
        "-0.5 / (x ** 5) * (-0.5 * (ca.sign(xdot) - 1)) * xdot ** 2"
    )
    collision_finsler: str = (
        "0.1/(x**1) * xdot**2"
    )
    limit_geometry: str = (
        "-0.1 / (x ** 1) * xdot ** 2"
    )
    limit_finsler: str = (
        "0.1/(x**1) * (-0.5 * (ca.sign(xdot) - 1)) * xdot**2"
    )
    self_collision_geometry: str = (
        "-0.5 / (x ** 1) * (-0.5 * (ca.sign(xdot) - 1)) * xdot ** 2"
    )
    self_collision_finsler: str = (
        "0.1/(x**1) * xdot**2"
    )
    attractor_potential: str = (
        "5.0 * (ca.norm_2(x) + 1 / 10 * ca.log(1 + ca.exp(-2 * 10 * ca.norm_2(x))))"
    )
    attractor_metric: str = (
        "((2.0 - 0.3) * ca.exp(-1 * (0.75 * ca.norm_2(x))**2) + 0.3) * ca.SX(np.identity(x.size()[0]))"
    )
    damper_beta: str = (
        "0.5 * (ca.tanh(-0.5 * (ca.norm_2(x) - 0.02)) + 1) * 6.5 + 0.01 + ca.fmax(0, sym('a_ex') - sym('a_le'))"
    )
    damper_eta: str = (
        "0.5 * (ca.tanh(-0.9 * (1 - 1/2) * ca.dot(xdot, xdot) - 0.5) + 1)"
    )
    """
    damper_beta: str = (
        "0.5 * (ca.tanh(-sym('alpha_b') * (ca.norm_2(x) - sym('radius_shift'))) + 1) * sym('beta_close') + sym('beta_distant') + ca.fmax(0, sym('a_ex') - sym('a_le'))"
    )
    damper_eta: str = (
        "0.5 * (ca.tanh(-sym('alpha_eta') * sym('ex_lag') * (1 - sym('ex_factor')) - 0.5) + 1)"
    )
    """
    urdf: str = None
    root_link: str = 'base_link'
    end_link: str = 'ee_link'


class ParameterizedFabricPlanner(object):
    def __init__(self, dof: int, robot_type: str, **kwargs):
        self._dof = dof
        self._config = FabricPlannerConfig(**kwargs)
        if self._config.urdf:
            self._forward_kinematics = GenericURDFFk(
                self._config.urdf,
                rootLink=self._config.root_link,
                end_link=self._config.end_link,
            )
        else:
            self._forward_kinematics = FkCreator(robot_type).fk()
        self.initialize_joint_variables()
        self.set_base_geometry()
        self._target_velocity = np.zeros(self._geometry.x().size()[0])
        self._ref_sign = 1
        self.leaves = {}

    """ INITIALIZING """

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
        self._forced_geometry += WeightedGeometry(
            g=geometry, le=lagrangian
        ).pull(forward_map)
        if prime_forcing_leaf:
            self._forced_variables = geometry._vars
            self._forced_forward_map = forward_map
        self._variables = self._variables + self._forced_geometry._vars

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

    def set_execution_energy(self, execution_lagrangian: Lagrangian):
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
        except AttributeError:
            logging.warning("No damping")

    def set_speed_control(self):
        self._geometry.concretize()
        self._forced_geometry.concretize(ref_sign=self._ref_sign)
        x_psi = self._forced_variables.position_variable()
        dm_psi = self._forced_forward_map
        exLag = self._execution_lagrangian
        a_ex = ca.SX.sym("a_ex", 1)
        a_le = ca.SX.sym("a_le", 1)
        beta_expression = self.config.damper_beta
        eta_expression = self.config.damper_eta
        self._damper = Damper(beta_expression, eta_expression, x_psi, dm_psi, exLag._l)
        self._variables.add_parameters(self._damper.symbolic_parameters())

    def get_forward_kinematics(self, link_name) -> ca.SX:
        if isinstance(link_name, ca.SX):
            return link_name
        if self._config.urdf:
            fk = self._forward_kinematics.fk(
                self._variables.position_variable(),
                self._config.root_link,
                link_name,
                positionOnly=True
            )
        else:
            fk = self._forward_kinematics.fk(
                self._variables.position_variable(),
                link_name,
                positionOnly=True
            )
        return fk


    """ DEFAULT COMPOSITION """
    def set_components(
        self,
        collision_links: list = None,
        self_collision_pairs: dict = None,
        collision_links_esdf: list = None,
        goal: GoalComposition = None,
        limits: list = None,
        number_obstacles: int = 1,
        number_dynamic_obstacles: int = 0,
        dynamic_obstacle_dimension: int = 3,
    ):
        if collision_links is None:
            collision_links = []
        if collision_links_esdf is None:
            collision_links_esdf = []
        if self_collision_pairs is None:
            self_collision_pairs = {}
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
                logging.warning(f"Expression {fk} for link {collision_link} is sparse and thus skipped.")
                continue
            for i in range(number_obstacles):
                obstacle_name = f"obst_{i}"
                geometry = ObstacleLeaf(self._variables, fk, obstacle_name, collision_link)
                geometry.set_geometry(self.config.collision_geometry)
                geometry.set_finsler_structure(self.config.collision_finsler)
                self.add_leaf(geometry)
            for i in range(number_dynamic_obstacles):
                obstacle_name = f"obst_dynamic_{i}"
                geometry = DynamicObstacleLeaf(self._variables, fk, obstacle_name, collision_link, reference_parameters=reference_parameter_list[i])
                geometry.set_geometry(self.config.collision_geometry)
                geometry.set_finsler_structure(self.config.collision_finsler)
                self.add_leaf(geometry)


        for collision_link in collision_links_esdf:
            fk = self.get_forward_kinematics(collision_link)
            geometry = ESDFGeometryLeaf(self._variables, collision_link, fk)
            geometry.set_geometry(self.config.collision_geometry)
            geometry.set_finsler_structure(self.config.collision_finsler)
            self.add_leaf(geometry)

        for self_collision_key, self_collision_list in self_collision_pairs.items():
            fk_key = self.get_forward_kinematics(self_collision_key)
            for self_collision_link in self_collision_list:
                fk_link = self.get_forward_kinematics(self_collision_link)
                fk = fk_link - fk_key
                if is_sparse(fk):
                    logging.warning(f"Expression {fk} for links {self_collision_key} and {self_collision_link} is sparse and thus skipped.")
                    continue
                self_collision_name = f"self_collision_{self_collision_key}_{self_collision_link}"
                geometry = SelfCollisionLeaf(self._variables, fk, self_collision_name)
                geometry.set_geometry(self.config.self_collision_geometry)
                geometry.set_finsler_structure(self.config.self_collision_finsler)
                self.add_leaf(geometry)

        if limits:
            for joint_index in range(len(limits)):
                lower_limit_geometry = LimitLeaf(self._variables, joint_index, limits[joint_index][0], 0)
                lower_limit_geometry.set_geometry(self.config.limit_geometry)
                lower_limit_geometry.set_finsler_structure(self.config.limit_finsler)
                upper_limit_geometry = LimitLeaf(self._variables, joint_index, limits[joint_index][1], 1)
                upper_limit_geometry.set_geometry(self.config.limit_geometry)
                upper_limit_geometry.set_finsler_structure(self.config.limit_finsler)
                self.add_leaf(lower_limit_geometry)
                self.add_leaf(upper_limit_geometry)

        execution_energy = ExecutionLagrangian(self._variables)
        self.set_execution_energy(execution_energy)
        if goal:
            self.set_goal_component(goal)
            # Adds default execution energy
            execution_energy = ExecutionLagrangian(self._variables)
            self.set_execution_energy(execution_energy)
            # Sets speed control
            self.set_speed_control()

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
                angles = ca.SX.sym(f"angle_goal_{sub_goal_index}", 3, 3)
                self._variables.add_parameter(f'angle_goal_{sub_goal_index}', angles)
                # rotation
                R = compute_rotation_matrix(angles)
                fk_child = ca.mtimes(R, fk_child)
                fk_parent = ca.mtimes(R, fk_parent)
            elif angles:
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
        try:
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
        except AttributeError:
            logging.warn("No forcing term, using pure geoemtry with energization.")
            self._geometry.concretize()
            #xddot = self._geometry._xddot - self._geometry._alpha * self._geometry._vars.velocity_variable()
            xddot = self._execution_geometry._xddot - self._execution_geometry._alpha * self._geometry._vars.velocity_variable()

        if mode == 'acc':
            self._funs = CasadiFunctionWrapper(
                "funs", self.variables.asDict(), {"action": xddot}
            )
        elif mode == 'vel':
            action = self._geometry.xdot() + time_step * xddot
            self._funs = CasadiFunctionWrapper(
                "funs", self.variables.asDict(), {"action": action}
            )
            

    def serialize(self, file_name: str):
        """
        Serializes the fabric planner.

        The file can be loaded using the serialized_planner.
        Essentially, only the casadiFunctionWrapper is serialized using
        pickle.
        """
        self.concretize()
        self._funs.serialize(file_name)
 
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
        #logging.debug(f"a_ex: {evaluations['a_ex']}")
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


