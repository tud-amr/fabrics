import casadi as ca
import numpy as np
import logging

from copy import deepcopy

from fabrics.diffGeometry.spec import Spec, checkCompatability
from fabrics.diffGeometry.diffMap import DifferentialMap, DynamicDifferentialMap

from fabrics.helpers.functions import joinRefTrajs
from fabrics.helpers.variables import Variables
from fabrics.helpers.casadiFunctionWrapper import CasadiFunctionWrapper

from fabrics.helpers.constants import eps

class LagrangianException(Exception):
    def __init__(self, expression, message):
        self._expression = expression
        self._message = message

    def what(self):
        return self._expression + ": " + self._message


class Lagrangian(object):
    """description"""

    def __init__(self, l: ca.SX, **kwargs):
        self._l = l
        self._x_ref_name = "x_ref"
        self._xdot_ref_name = "xdot_ref"
        self._xddot_ref_name = "xddot_ref"
        assert isinstance(l, ca.SX)
        if 'x' in kwargs:
            self._vars = Variables(state_variables={"x": kwargs.get('x'), "xdot": kwargs.get('xdot')})
        elif 'var' in kwargs:
            self._vars = kwargs.get('var')
        self._rel = False
        self._refTrajs = []
        if 'ref_names' in kwargs:
            ref_names = kwargs.get('ref_names')
            self._x_ref_name = ref_names[0]
            self._xdot_ref_name = ref_names[1]
            self._xddot_ref_name = ref_names[2]
        if 'refTrajs' in kwargs:
            self._refTrajs = kwargs.get('refTrajs')
            self._rel = len(self._refTrajs) > 0
        if self.is_dynamic():
            self._J_ref_inv = np.identity(self.x_ref().size()[0])
        if "J_ref" in kwargs:
            self._J_ref = kwargs.get("J_ref")
            logging.warning("Casadi pseudo inverse is used in Lagrangian")
            self._J_ref_inv = ca.mtimes(ca.transpose(self._J_ref), ca.inv(ca.mtimes(self._J_ref, ca.transpose(self._J_ref)) + np.identity(self.x_ref().size()[0]) * eps))
        self.applyEulerLagrange()


    def x_ref(self):
        return self._vars.parameter_by_name(self._x_ref_name)

    def xdot_ref(self):
        return self._vars.parameter_by_name(self._xdot_ref_name)

    def x(self):
        return self._vars.position_variable()

    def xdot(self):
        return self._vars.velocity_variable()

    def xdot_rel(self, ref_sign: int = 1):
        if self.is_dynamic():
            return self.xdot() - ca.mtimes(self._J_ref_inv, self.xdot_ref()) * ref_sign
        else:
            return self.xdot()

    def __add__(self, b):
        assert isinstance(b, Lagrangian)
        checkCompatability(self, b)
        refTrajs = joinRefTrajs(self._refTrajs, b._refTrajs)
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
        new_vars = self._vars + b._vars
        return Lagrangian(self._l + b._l, var=new_vars, **ref_arguments)

    def is_dynamic(self) -> bool:
        logging.debug(f"Lagrangian is dynamic: {self._x_ref_name in self._vars.parameters()}")
        return self._x_ref_name in self._vars.parameters()


    def applyEulerLagrange(self):
        dL_dxdot = ca.gradient(self._l, self.xdot())
        dL_dx = ca.gradient(self._l, self.x())
        d2L_dxdxdot = ca.jacobian(dL_dx, self.xdot())
        d2L_dxdot2 = ca.jacobian(dL_dxdot, self.xdot())
        f_rel = np.zeros(self.x().size()[0])
        en_rel = np.zeros(1)

        if self.is_dynamic():
            x_ref = self._vars.parameters()[self._x_ref_name]
            xdot_ref = self._vars.parameters()[self._xdot_ref_name]
            xddot_ref = self._vars.parameters()[self._xddot_ref_name]
            dL_dxpdot = ca.gradient(self._l, xdot_ref)
            d2L_dxdotdxpdot = ca.jacobian(dL_dxdot, xdot_ref)
            d2L_dxdotdxp = ca.jacobian(dL_dxdot, x_ref)
            f_rel1 = ca.mtimes(d2L_dxdotdxpdot, xddot_ref)
            f_rel2 = ca.mtimes(d2L_dxdotdxp, xdot_ref)
            f_rel += f_rel1 + f_rel2
            en_rel += ca.dot(dL_dxpdot, xdot_ref)

        F = d2L_dxdxdot
        f_e = -dL_dx
        M = d2L_dxdot2
        f = ca.mtimes(ca.transpose(F), self.xdot()) + f_e + f_rel
        self._H = ca.dot(dL_dxdot, self.xdot()) - self._l + en_rel
        self._S = Spec(M, f=f, var=self._vars, refTrajs=self._refTrajs)

    def concretize(self):
        self._S.concretize()
        var = deepcopy(self._vars)
        for refTraj in self._refTrajs:
            var += refTraj._vars
        self._funs = CasadiFunctionWrapper(
            "funs", var.asDict(), {"H": self._H}
        )

    def ref_names(self) -> list:
        return [self._x_ref_name, self._xdot_ref_name, self._xddot_ref_name]


    def evaluate(self, **kwargs):
        funs = self._funs.evaluate(**kwargs)
        H = funs['H']
        M, f, _ = self._S.evaluate(**kwargs)
        return M, f, H

    def pull(self, dm: DifferentialMap):
        assert isinstance(dm, DifferentialMap)
        l_subst = ca.substitute(self._l, self.x(), dm._phi)
        l_subst2 = ca.substitute(l_subst, self.xdot(), dm.phidot())
        new_state_variables = dm.state_variables()
        new_parameters = {}
        new_parameters.update(self._vars.parameters())
        new_parameters.update(dm.params())
        new_vars = Variables(state_variables=new_state_variables, parameters=new_parameters)
        if hasattr(dm, '_refTraj'):
            refTrajs = [dm._refTraj] + [refTraj.pull(dm) for refTraj in self._refTrajs]
        else:
            refTrajs = [refTraj.pull(dm) for refTraj in self._refTrajs]
        J_ref = dm._J
        if self.is_dynamic():
            return Lagrangian(l_subst2, var=new_vars, J_ref=J_ref, ref_names=self.ref_names())
        else:
            return Lagrangian(l_subst2, var=new_vars, ref_names=self.ref_names())

    def dynamic_pull(self, dm: DynamicDifferentialMap):
        l_pulled = self._l
        l_pulled_subst_x = ca.substitute(l_pulled, self.x(), dm._phi)
        l_pulled_subst_x_xdot = ca.substitute(l_pulled_subst_x, self.xdot(), dm.phidot())
        return Lagrangian(l_pulled_subst_x_xdot, var=dm._vars, ref_names=dm.ref_names())


class FinslerStructure(Lagrangian):
    def __init__(self, lg: ca.SX, **kwargs):
        self._lg = lg
        l = 0.5 * lg ** 2
        super().__init__(l, **kwargs)

    def concretize(self):
        super().concretize()
        self._funs_lg = CasadiFunctionWrapper(
            "funs", self._vars.asDict(), {"Lg": self._lg}
        )

    def evaluate(self, **kwargs):
        M, f, l = super().evaluate(**kwargs)
        lg = self._funs_lg.evaluate(**kwargs)['Lg']
        return M, f, l, lg
