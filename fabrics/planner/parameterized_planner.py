from dataclasses import dataclass, field
from typing import Dict
import casadi as ca
import numpy as np
from copy import deepcopy

from fabrics.helpers.variables import Variables

from fabrics.diffGeometry.diffMap import DifferentialMap, ParameterizedDifferentialMap
from fabrics.diffGeometry.energy import Lagrangian
from fabrics.diffGeometry.geometry import Geometry
from fabrics.diffGeometry.energized_geometry import WeightedGeometry
from fabrics.diffGeometry.speedControl import Damper
from fabrics.helpers.functions import joinRefTrajs

from fabrics.helpers.variables import Variables
from fabrics.helpers.casadiFunctionWrapper import CasadiFunctionWrapper

from fabrics.leaves.leaf import Leaf
from fabrics.leaves.attractor import Attractor, ParameterizedAttractor
from fabrics.leaves.obstacle_leaf import ObstacleLeaf
from fabrics.planner.default_energies import ExecutionLagrangian

from fabrics.leaves.generics.attractor import GenericAttractor
from fabrics.leaves.generics.geometry import GenericGeometry
from fabrics.leaves.generics.geometry import ObstacleGeometry


@dataclass
class FabricPlannerConfig:
    base_inertia: float = 0.2
    #s = -0.5 * (ca.sign(xdot) - 1)
    #h = -p["lam"] / (x ** p["exp"]) * s * xdot ** 2
    collision_geometry: str = (
        "-2 / (x_geometry ** 2) * (-0.5 * (ca.sign(xdot_geometry) - 1)) * xdot_geometry ** 2"
    )
    collision_finsler: str = (
        "2.0/(x_geometry**1) * xdot_geometry**2"
    )
    self_collision_geometry: str = (
        "-0.5 * / (x_self ** 2) * (-0.5 * (ca.sign(xdot_self) - 1) * xdot_self ** 2"
    )
    attractor_potential: str = (
        "5.0 * (ca.norm_2(x_goal) + 1 / 10 * ca.log(1 + ca.exp(-2 * 10 * ca.norm_2(x_goal))))"
    )
    attractor_metric: str = (
        "((2.0 - 0.3) * ca.exp(-1 * (0.75 * ca.norm_2(x_goal))**2) + 0.3) * ca.SX(np.identity(x_goal.size()[0]))"
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


class ParameterizedFabricPlanner(object):
    def __init__(self, dof: int, **kwargs):
        self._dof = dof
        self._config = FabricPlannerConfig(**kwargs)
        self.initialize_joint_variables()
        self.set_base_geometry()

    """ INITIALIZING """

    def initialize_joint_variables(self):
        q = ca.SX.sym("q", self._dof)
        qdot = ca.SX.sym("qdot", self._dof)
        self._variables = Variables(state_variables={"q": q, "qdot": qdot})

    def set_base_geometry(self):
        qdot = self._variables.velocity_variable()
        base_energy = 0.5 * self._config.base_inertia * ca.dot(qdot, qdot)
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

    def add_leaf(self, leaf: Leaf) -> None:
        if isinstance(leaf, GenericAttractor):
            self.add_forcing_geometry(leaf.map(), leaf.lagrangian(), leaf.geometry())
        if isinstance(leaf, GenericGeometry):
            self.add_geometry(leaf.map(), leaf.lagrangian(), leaf.geometry())

    def add_forcing_geometry(
        self,
        forward_map: DifferentialMap,
        lagrangian: Lagrangian,
        geometry: Geometry,
    ) -> None:
        assert isinstance(forward_map, ParameterizedDifferentialMap)
        assert isinstance(lagrangian, Lagrangian)
        assert isinstance(geometry, Geometry)
        self._forced_geometry = deepcopy(self._geometry)
        self._forced_geometry += WeightedGeometry(
            g=geometry, le=lagrangian
        ).pull(forward_map)
        self._forced_variables = geometry._vars
        self._forced_forward_map = forward_map
        self._variables = self._variables + self._forced_geometry._vars

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

    """ DEFAULT COMPOSITION """
    def set_components(self, fks, fk_goal, number_obstacles = 1, goal = True):
        # Adds default obstacle
        obst_dimension = fks[0].size()[0]
        for i in range(number_obstacles):
            obstacle_name = f"obst_{i}"
            for fk in fks:
                geometry = ObstacleGeometry(self._variables, fk, obstacle_name)
                geometry.set_geometry(self.config.collision_geometry)
                geometry.set_finsler_structure(self.config.collision_finsler)
                self.add_leaf(geometry)
        if goal:
            # Adds default attractor
            goal_dimension = fk_goal.size()[0]
            self._variables.add_parameter('x_goal', ca.SX.sym('x_goal', goal_dimension))
            attractor = GenericAttractor(self._variables, fk_goal)
            attractor.set_potential(self.config.attractor_potential)
            attractor.set_metric(self.config.attractor_metric)
            self.add_leaf(attractor)
            # Adds default execution energy
            execution_energy = ExecutionLagrangian(self._variables)
            self.set_execution_energy(execution_energy)
            # Sets speed control
            self.set_speed_control()


    def concretize(self):
        try:
            a_ex = (
                self._eta * self._geometry._alpha
                + (1 - self._eta) * self._forced_geometry._alpha
            )
            beta_subst = self._beta.substitute(-a_ex, -self._geometry._alpha)
            xddot = self._forced_geometry._xddot - (a_ex + beta_subst) * self._geometry.xdot()
        except AttributeError:
            print("No forcing term, using pure geoemtry")
            self._geometry.concretize()
            xddot = self._geometry._xddot - self._geometry._alpha * self._geometry._vars.velocity_variable()
        self._funs = CasadiFunctionWrapper(
            "funs", self.variables.asDict(), {"xddot": xddot}
        )


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

    def __del__(self):
        del(self._variables)
        print("PLANNER DELETED")


