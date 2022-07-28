import pdb
import casadi as ca
import numpy as np
import logging

from forwardkinematics.fksCommon.fk_creator import FkCreator

from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner, FabricPlannerConfig
from fabrics.helpers.casadiFunctionWrapper import CasadiFunctionWrapper


class NonHolonomicFabricPlannerConfig(FabricPlannerConfig):
    l_offset: float = 0.2
    m_arm: float = 1.0
    m_base: float= 1.0
    m_rot: float = 1.0

class NonHolonomicParameterizedFabricPlanner(ParameterizedFabricPlanner):
    def __init__(
        self,
        dof: int,
        robot_type: str,
        **kwargs
    ):
        self._dof = dof
        self._config = NonHolonomicFabricPlannerConfig(**kwargs)
        self._forward_kinematics = FkCreator(robot_type).fk()
        self.initialize_joint_variables()
        self.set_base_geometry()
        self._target_velocity = np.zeros(self._geometry.x().size()[0])
        self._ref_sign = 1

        M = np.identity(self._dof)
        M[0:2, 0:2] *= self._config.m_base
        M[2, 2] *= self._config.m_rot
        M[3:self._dof, 3:self._dof] *= self._config.m_arm
        q = self._variables.position_variable()
        qudot = ca.SX.sym("qudot", self._dof - 1)
        J_nh = ca.SX(np.zeros((self._dof, self._dof-1)))

        J_nh[0, 0] = ca.cos(q[2])
        J_nh[0, 1] = -self._config.l_offset * ca.sin(q[2])
        J_nh[1, 0] = ca.sin(q[2])
        J_nh[1, 1] = self._config.l_offset * ca.cos(q[2])
        for i in range(2, self._dof):
            J_nh[i, i-1] = 1
        f_extra = ca.SX(np.zeros((self._dof, 1)))
        f_extra[0] = qudot[0] * qudot[1] * -ca.sin(q[2]) - self._config.l_offset * ca.cos(q[2]) * qudot[1]**2
        f_extra[1] = qudot[0] * qudot[1] * ca.sin(q[2]) - self._config.l_offset * ca.sin(q[2]) * qudot[1]**2
        self._J_nh = J_nh
        self._qudot = qudot
        self._variables.add_state_variable('qudot', qudot)
        self._f_extra = f_extra

    def extra_terms_function(self):
        extra_terms_functions = CasadiFunctionWrapper("extra_terms", self._variables.asDict(), {"J_nh": self._J_nh, "f_extra": self._f_extra})
        return extra_terms_functions

    def concretize(self):
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
        self._funs = CasadiFunctionWrapper(
            "funs", self.variables.asDict(), {"xddot": xddot}
        )
