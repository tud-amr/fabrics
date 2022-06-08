from dataclasses import dataclass, field
from typing import Dict
import casadi as ca
import numpy as np
from copy import deepcopy
from fabrics.helpers.exceptions import ExpressionSparseError

from fabrics.helpers.variables import Variables
from fabrics.helpers.functions import is_sparse, parse_symbolic_input

from fabrics.diffGeometry.diffMap import DifferentialMap, ParameterizedDifferentialMap, DynamicParameterizedDifferentialMap
from fabrics.diffGeometry.energy import Lagrangian
from fabrics.diffGeometry.geometry import Geometry
from fabrics.diffGeometry.energized_geometry import WeightedGeometry
from fabrics.diffGeometry.speedControl import Damper

from fabrics.helpers.casadiFunctionWrapper import CasadiFunctionWrapper

from fabrics.components.energies.execution_energies import ExecutionLagrangian
from fabrics.components.leaves.leaf import Leaf
from fabrics.components.leaves.attractor import GenericAttractor
from fabrics.components.leaves.dynamic_attractor import GenericDynamicAttractor
from fabrics.components.leaves.geometry import GenericGeometryLeaf
from fabrics.components.leaves.geometry import ObstacleLeaf, LimitLeaf, SelfCollisionLeaf

from MotionPlanningGoal.goalComposition import GoalComposition
from MotionPlanningGoal.subGoal import SubGoal

from forwardkinematics.fksCommon.fk_creator import FkCreator
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

from pyquaternion import Quaternion

class InvalidRotationAnglesError(Exception):
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
        "-0.01 / (x ** 8) * (-0.5 * (ca.sign(xdot) - 1)) * xdot ** 2"
    )
    limit_finsler: str = (
        "0.01/(x**1) * xdot**2"
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
    ex_factor: float = 1.0
    damper: Dict[str, float] = field(
        default_factory=lambda: (
            {
                "alpha_b": 0.5,
                "alpha_eta": 0.5,
                "alpha_shift": 0.5,
                "beta_distant": 0.01,
                "beta_close": 6.5,
                "radius_shift": 0.02,
            }
        )
    )
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
        if isinstance(leaf, GenericDynamicAttractor):
            self.add_dynamic_forcing_geometry(leaf.map(), leaf.dynamic_map(), leaf.lagrangian(), leaf.geometry(), leaf._xdot_ref, prime_leaf)
        if isinstance(leaf, GenericGeometryLeaf):
            self.add_geometry(leaf.map(), leaf.lagrangian(), leaf.geometry())

    def add_forcing_geometry(
        self,
        forward_map: DifferentialMap,
        lagrangian: Lagrangian,
        geometry: Geometry,
        prime_forcing_leaf: bool,
    ) -> None:
        assert isinstance(forward_map, ParameterizedDifferentialMap)
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
        assert isinstance(dynamic_map, DynamicParameterizedDifferentialMap)
        assert isinstance(lagrangian, Lagrangian)
        assert isinstance(geometry, Geometry)
        assert isinstance(target_velocity, ca.SX)
        if not hasattr(self, '_forced_geometry'):
            self._forced_geometry = deepcopy(self._geometry)
        self._forced_geometry += WeightedGeometry(
            g=geometry, le=lagrangian
        ).pull(dynamic_map).pull(forward_map)
        if prime_forcing_leaf:
            self._forced_variables = geometry._vars
            self._forced_forward_map = forward_map
        self._variables = self._variables + self._forced_geometry._vars
        self._target_velocity += ca.mtimes(ca.transpose(forward_map._J), target_velocity)

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
            #self._forced_speed_controlled_geometry.concretize()
        except AttributeError:
            print("No damping")

    def set_speed_control(self):
        self._geometry.concretize()
        self._forced_geometry.concretize()
        alpha_b = self.config.damper["alpha_b"]
        alpha_eta = self.config.damper["alpha_eta"]
        alpha_shift = self.config.damper["alpha_shift"]
        radius_shift = self.config.damper["radius_shift"]
        beta_distant = self.config.damper["beta_distant"]
        beta_close = self.config.damper["beta_close"]
        ex_factor = self.config.ex_factor
        x_psi = self._forced_variables.position_variable()
        dm_psi = self._forced_forward_map
        exLag = self._execution_lagrangian
        ex_factor = self.config.ex_factor
        s_beta = 0.5 * (ca.tanh(-alpha_b * (ca.norm_2(x_psi) - radius_shift)) + 1)
        a_ex = ca.SX.sym("a_ex", 1)
        a_le = ca.SX.sym("a_le", 1)
        beta_fun = s_beta * beta_close + beta_distant + ca.fmax(0, a_ex - a_le)
        self._beta = Damper(beta_fun, a_ex, a_le, x_psi, dm_psi)
        l_ex_d = ex_factor * exLag._l
        self._eta = 0.5 * (ca.tanh(-alpha_eta * (exLag._l - l_ex_d) - alpha_shift) + 1)

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
        collision_links: list,
        self_collision_pairs: dict,
        goal: GoalComposition = None,
        limits: list = None,
        number_obstacles: int = 1
    ):
        # Adds default obstacle avoidance
        for collision_link in collision_links:
            fk = self.get_forward_kinematics(collision_link)
            if is_sparse(fk):
                print(f"Expression {fk} for link {collision_link} is sparse and thus skipped.")
                continue
            for i in range(number_obstacles):
                obstacle_name = f"obst_{i}"
                geometry = ObstacleLeaf(self._variables, fk, obstacle_name, collision_link)
                geometry.set_geometry(self.config.collision_geometry)
                geometry.set_finsler_structure(self.config.collision_finsler)
                self.add_leaf(geometry)
        for self_collision_key, self_collision_list in self_collision_pairs.items():
            fk_key = self.get_forward_kinematics(self_collision_key)
            for self_collision_link in self_collision_list:
                fk_link = self.get_forward_kinematics(self_collision_link)
                fk = fk_link - fk_key
                if is_sparse(fk):
                    print(f"Expression {fk} for links {self_collision_key} and {self_collision_link} is sparse and thus skipped.")
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
            fk_child = self.get_forward_kinematics(sub_goal.childLink())
            fk_parent = self.get_forward_kinematics(sub_goal.parentLink())
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
            for j, sub_goal in enumerate(goal.subGoals()):
                fk_sub_goal = self.get_differential_map(j, sub_goal)
                goal_dimension = sub_goal.m()
                if is_sparse(fk_sub_goal):
                    raise ExpressionSparseError()
                if sub_goal.type() == 'analyticSubGoal':
                    attractor = GenericDynamicAttractor(self._variables, fk_sub_goal, f"goal_{j}")
                else:
                    self._variables.add_parameter(f'x_goal_{j}', ca.SX.sym(f'x_goal_{j}', goal_dimension))
                    attractor = GenericAttractor(self._variables, fk_sub_goal, f"goal_{j}")
                attractor.set_potential(self.config.attractor_potential)
                attractor.set_metric(self.config.attractor_metric)
                self.add_leaf(attractor, prime_leaf=sub_goal.isPrimeGoal())


    def concretize(self):
        try:
            a_ex = (
                self._eta * self._geometry._alpha
                + (1 - self._eta) * self._forced_geometry._alpha
            )
            beta_subst = self._beta.substitute(-a_ex, -self._geometry._alpha)
            xddot = self._forced_geometry._xddot - (a_ex + beta_subst) * (
                self._geometry.xdot()
                - ca.mtimes(self._forced_geometry.Minv(), self._target_velocity)
            )
            #xddot = self._forced_geometry._xddot
        except AttributeError:
            print("No forcing term, using pure geoemtry")
            self._geometry.concretize()
            xddot = self._geometry._xddot - self._geometry._alpha * self._geometry._vars.velocity_variable()
        self._funs = CasadiFunctionWrapper(
            "funs", self.variables.asDict(), {"xddot": xddot}
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
        """
        evaluations = self._funs.evaluate(**kwargs)
        action = evaluations["xddot"]
        """
        # avoid to small actions
        if np.linalg.norm(action) < eps:
            action = np.zeros(self._n)
        """
        return action

    """
    def __del__(self):
        del(self._variables)
        print("PLANNER DELETED")
    """


