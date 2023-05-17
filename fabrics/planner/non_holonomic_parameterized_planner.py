from dataclasses import dataclass
import casadi as ca
import numpy as np
import logging

from forwardkinematics.fksCommon.fk_creator import FkCreator
from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner, FabricPlannerConfig
from fabrics.diffGeometry.energy import Lagrangian
from fabrics.diffGeometry.geometry import Geometry
from fabrics.diffGeometry.energized_geometry import WeightedGeometry
from fabrics.helpers.casadiFunctionWrapper import CasadiFunctionWrapper
from fabrics.helpers.functions import parse_symbolic_input


@dataclass
class NonHolonomicFabricPlannerConfig(FabricPlannerConfig):
    l_offset: str = "0.2"
    M_base_energy: str = (
        "ca.hcat([ca.vcat([sym('m_base_x'),0,0]), ca.vcat([0,sym('m_base_y'),0]), ca.vcat([0, 0, sym('m_rot')])])"
    )
    M_arm_energy: str = (
        "sym('m_arm') * np.identity(x.size()[0] - 3)"
    )

class NonHolonomicParameterizedFabricPlanner(ParameterizedFabricPlanner):
    def __init__(
        self,
        dof: int,
        robot_type: str,
        facing_direction: str = '-y',
        **kwargs
    ):
        self.leaves = {}
        self._dof = dof
        self._config = NonHolonomicFabricPlannerConfig(**kwargs)
        if self._config.urdf:
            self._forward_kinematics = GenericURDFFk(
                self._config.urdf,
                rootLink=self._config.root_link,
                end_link=self._config.end_link,
                base_type='diffdrive',
            )
        else:
            self._forward_kinematics = FkCreator(robot_type).fk()
        self.initialize_joint_variables()
        self.set_base_geometry()
        self._target_velocity = np.zeros(self._geometry.x().size()[0])
        self._ref_sign = 1
        self.set_non_holonomic_constraints(facing_direction=facing_direction)


    def set_non_holonomic_constraints(self, facing_direction: str = '-y'):
        q = self._variables.position_variable()
        qdot = self._variables.velocity_variable()
        qudot = ca.SX.sym("qudot", self._dof - 1)
        new_parameters, l_offset = parse_symbolic_input(self._config.l_offset, q, qdot)
        self._variables.add_parameters(new_parameters)
        J_nh = ca.SX(np.zeros((self._dof, self._dof-1)))
        for i in range(2, self._dof):
            J_nh[i, i-1] = 1
        f_extra = ca.SX(np.zeros((self._dof, 1)))
        if facing_direction == '-y':
            J_nh[0, 0] = ca.sin(q[2])
            J_nh[0, 1] = l_offset * ca.cos(q[2])
            J_nh[1, 0] = -ca.cos(q[2])
            J_nh[1, 1] = l_offset * ca.sin(q[2])
            f_extra[0] = qudot[0] * qudot[1] * ca.cos(q[2]) - l_offset * ca.sin(q[2]) * qudot[1]**2
            f_extra[1] = qudot[0] * qudot[1] * ca.sin(q[2]) + l_offset * ca.cos(q[2]) * qudot[1]**2
        elif facing_direction == 'x':
            logging.warning("Not sure about this, yet.")
            J_nh[0, 0] = ca.cos(q[2])
            J_nh[0, 1] = -l_offset * ca.sin(q[2])
            J_nh[1, 0] = ca.sin(q[2])
            J_nh[1, 1] = l_offset * ca.cos(q[2])
            f_extra[0] = qudot[0] * qudot[1] * -ca.sin(q[2]) - l_offset * ca.cos(q[2]) * qudot[1]**2
            f_extra[1] = qudot[0] * qudot[1] * ca.sin(q[2]) - l_offset * ca.sin(q[2]) * qudot[1]**2
        self._J_nh = J_nh
        self._f_extra = f_extra
        self._qudot = qudot
        self._variables.add_state_variable('qudot', qudot)


    def set_base_geometry(self):
        q = self._variables.position_variable()
        qdot = self._variables.velocity_variable()
        new_parameters, M_base_energy =  parse_symbolic_input(self._config.M_base_energy, q, qdot)
        self._variables.add_parameters(new_parameters)
        new_parameters, M_arm_energy =  parse_symbolic_input(self._config.M_arm_energy, q, qdot)
        self._variables.add_parameters(new_parameters)
        M_base = ca.SX(np.identity(q.size()[0]))
        M_base[0:3,0:3] = M_base_energy
        M_base[3:q.size()[0],3:q.size()[0]] = M_arm_energy
        base_energy = 0.5 * ca.dot(qdot, ca.mtimes(M_base, qdot))
        base_geometry = Geometry(h=ca.SX(np.zeros(self._dof)), var=self.variables)
        base_lagrangian = Lagrangian(base_energy, var=self._variables)
        self._geometry = WeightedGeometry(g=base_geometry, le=base_lagrangian)

    def extra_terms_function(self):
        extra_terms_functions = CasadiFunctionWrapper("extra_terms", self._variables.asDict(), {"J_nh": self._J_nh, "f_extra": self._f_extra})
        return extra_terms_functions

    def concretize(self, mode='acc', time_step=None):
        if mode == 'vel':
            if not time_step:
                raise Exception("No time step passed in velocity mode.")
        eps = 1e-6
        MJ = ca.mtimes(self._forced_geometry._M, self._J_nh)
        MJtMJ = ca.mtimes(ca.transpose(MJ), MJ) + ca.SX(np.identity(self._dof - 1)) * eps
        MJ_pinv = ca.mtimes(ca.inv(MJtMJ), ca.transpose(MJ))
        try:
            eta = self._damper.substitute_eta()
            a_ex = (
                eta * self._geometry._alpha
                + (1 - eta) * self._forced_geometry._alpha
            )
            beta_subst = self._damper.substitute_beta(-a_ex, -self._geometry._alpha)
            """
            xddot = self._forced_geometry._xddot - (a_ex + beta_subst) * (
                self._geometry.xdot()
                - ca.mtimes(self._forced_geometry.Minv(), self._target_velocity)
            )
            """
            xddot = ca.mtimes(
                MJ_pinv,
                - self._forced_geometry.f()
                - ca.mtimes(self._forced_geometry._M, self._f_extra)
                - ca.mtimes(self._forced_geometry._M, a_ex * self._forced_geometry.xdot())
            ) - beta_subst * self._qudot
            #xddot = ca.mtimes(MJ_pinv, -self._forced_geometry.f())
            #xddot = self._forced_geometry._xddot
        except AttributeError as e:
            logging.info("No forcing term, using pure geoemtry")
            raise AttributeError(e)
            logging.error(e)
            self._geometry.concretize()
            xddot = self._geometry._xddot - self._geometry._alpha * self._geometry._vars.velocity_variable()
        if mode == 'acc':
            self._funs = CasadiFunctionWrapper(
                "funs", self.variables.asDict(), {"action": xddot}
            )
        elif mode == 'vel':
            action = self._qudot + time_step * xddot
            self._funs = CasadiFunctionWrapper(
                "funs", self.variables.asDict(), {"action": action}
            )
