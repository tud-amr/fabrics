import casadi as ca
import numpy as np
import logging
from copy import deepcopy

from fabrics.diffGeometry.diffMap import DifferentialMap, DynamicDifferentialMap
from fabrics.helpers.constants import eps
from fabrics.helpers.functions import joinVariables, checkCompatability

from fabrics.helpers.casadiFunctionWrapper import CasadiFunctionWrapper
from fabrics.helpers.variables import Variables

class Spec:
    """description"""

    def __init__(self, M: ca.SX, **kwargs):
        self._x_ref_name = "x_ref"
        self._xdot_ref_name = "xdot_ref"
        self._xddot_ref_name = "xddot_ref"
        if 'ref_names' in kwargs:
            ref_names = kwargs.get('ref_names')
            self._x_ref_name = ref_names[0]
            self._xdot_ref_name = ref_names[1]
            self._xddot_ref_name = ref_names[2]
        if 'f' in kwargs:
            f = kwargs.get('f')
            assert isinstance(f, ca.SX)
            self._f = f
        if 'h' in kwargs:
            h = kwargs.get('h')
            assert isinstance(h, ca.SX)
            self._h = h
        if 'x' in kwargs:
            self._vars = Variables(state_variables={"x": kwargs.get('x'), "xdot": kwargs.get('xdot')})
        elif 'var' in kwargs:
            self._vars = kwargs.get('var')
        self._refTrajs = []
        if 'refTrajs' in kwargs:
            self._refTrajs = kwargs.get('refTrajs')
        if self.is_dynamic():
            self._J_ref = np.identity(self.x_ref().size()[0])
            self._J_ref_inv = np.identity(self.x_ref().size()[0])
        if "J_ref" in kwargs:
            self._J_ref = kwargs.get("J_ref")
            logging.warning("Casadi pseudo inverse is used in Lagrangian")
            self._J_ref_inv = ca.mtimes(ca.transpose(self._J_ref), ca.inv(ca.mtimes(self._J_ref, ca.transpose(self._J_ref)) + np.identity(self.x_ref().size()[0]) * eps))
        self._xdot_d = np.zeros(self.x().size()[0])
        self._vars.verify()
        assert isinstance(M, ca.SX)
        self._M = M

    def x_ref(self):
        return self._vars.parameter_by_name(self._x_ref_name)

    def h(self):
        if hasattr(self, '_h'):
            return self._h
        else:
            return ca.mtimes(self.Minv(), self._f)

    def ref_names(self) -> list:
        return [self._x_ref_name, self._xdot_ref_name, self._xddot_ref_name]

    def f(self):
        if hasattr(self, '_f'):
            return self._f
        else:
            return ca.mtimes(self.M(), self._h)

    def M(self):
        return self._M

    def Minv(self):
        logging.warning("Casadi pseudo inverse is used in spec")
        return ca.pinv(self._M + np.identity(self.x().size()[0]) * eps)

    def x(self):
        return self._vars.position_variable()

    def xdot(self):
        return self._vars.velocity_variable()

    def concretize(self):
        self._xddot = -self.h()
        var = deepcopy(self._vars)
        for refTraj in self._refTrajs:
            var += refTraj._vars
        self._funs = CasadiFunctionWrapper(
            "funs", var.asDict(), {"M": self.M(), "f": self.f(), "xddot": self._xddot}
        )

    def evaluate(self, **kwargs):
        evaluations = self._funs.evaluate(**kwargs)
        M_eval = evaluations["M"]
        if len(M_eval.shape) == 1:
            M_eval = np.array([M_eval])
        f_eval = evaluations["f"]
        xddot_eval = evaluations["xddot"]
        return [M_eval, f_eval, xddot_eval]

    def __add__(self, b):
        assert isinstance(b, Spec)
        checkCompatability(self, b)
        all_vars = self._vars + b._vars
        ref_names = []
        if self.is_dynamic():
            ref_names += self.ref_names()
        if b.is_dynamic():
            ref_names += b.ref_names()
        if len(ref_names) == 0:
            ref_names = ['x_ref', 'xdot_ref', 'xddot_ref']
        ref_names = []
        if self.is_dynamic():
            ref_names += self.ref_names()
            J_ref = self._J_ref
        if b.is_dynamic():
            ref_names += b.ref_names()
            J_ref = b._J_ref
        if len(ref_names) > 0:
            ref_arguments = {'ref_names': ref_names, 'J_ref': J_ref}
        else:
            ref_arguments = {}
        if hasattr(self, '_h') and hasattr(b, '_h'):
            return Spec(self.M() + b.M(), h=self.h() + b.h(), var=all_vars, **ref_arguments)
        else:
            return Spec(self.M() + b.M(), f=self.f() + b.f(), var=all_vars, **ref_arguments)

    def pull(self, dm: DifferentialMap):
        assert isinstance(dm, DifferentialMap)
        M_pulled = ca.mtimes(ca.transpose(dm._J), ca.mtimes(self.M(), dm._J))
        Jt = ca.transpose(dm._J)
        f_1 = ca.mtimes(Jt, ca.mtimes(self.M(), dm.Jdotqdot()))
        f_2 = ca.mtimes(Jt, self.f())
        f_pulled = f_1 + f_2
        x = self._vars.position_variable()
        xdot = self._vars.velocity_variable()
        M_pulled_subst_x = ca.substitute(M_pulled, x, dm._phi)
        M_pulled_subst_x_xdot = ca.substitute(
            M_pulled_subst_x, xdot, dm.phidot()
        )
        f_pulled_subst_x = ca.substitute(f_pulled, x, dm._phi)
        f_pulled_subst_x_xdot = ca.substitute(
            f_pulled_subst_x, xdot, dm.phidot()
        )
        new_state_variables = dm.state_variables()
        new_parameters = {}
        new_parameters.update(self._vars.parameters())
        new_parameters.update(dm.params())
        new_vars = Variables(state_variables=new_state_variables, parameters=new_parameters)
        J_ref = dm._J
        if self.is_dynamic():
            return Spec(M_pulled_subst_x_xdot, f=f_pulled_subst_x_xdot, var=new_vars, J_ref=J_ref, ref_names=self.ref_names())
        else:
            return Spec(M_pulled_subst_x_xdot, f=f_pulled_subst_x_xdot, var=new_vars, ref_names=self.ref_names())
        """
        if hasattr(dm, '_refTraj'):
            refTrajs = [dm._refTraj] + [refTraj.pull(dm) for refTraj in self._refTrajs]
        else:
            refTrajs = [refTraj.pull(dm) for refTraj in self._refTrajs]
        return Spec(M_pulled_subst_x_xdot, f=f_pulled_subst_x_xdot, var=new_vars, refTrajs=refTrajs)
        """

    def is_dynamic(self) -> bool:
        logging.debug(f"{self._x_ref_name}, {self._vars.parameters()}")
        return self._x_ref_name in self._vars.parameters()

    def dynamic_pull(self, dm: DynamicDifferentialMap):
        M_pulled = self.M()
        x = self._vars.position_variable()
        xdot = self._vars.velocity_variable()
        M_pulled_subst_x = ca.substitute(M_pulled, x, dm._phi)
        M_pulled_subst_x_xdot = ca.substitute(M_pulled_subst_x, xdot, dm.phidot())
        f_pulled = self.f()  - ca.mtimes(self.M(), dm.xddot_ref())
        f_pulled_subst_x = ca.substitute(f_pulled, x, dm._phi)
        f_pulled_subst_x_xdot = ca.substitute(
            f_pulled_subst_x, xdot, dm.phidot()
        )
        return Spec(M_pulled_subst_x_xdot, f=f_pulled_subst_x_xdot, var=dm._vars, ref_names=dm.ref_names())
