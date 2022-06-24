import pdb
import casadi as ca
import numpy as np

from copy import deepcopy

from fabrics.diffGeometry.spec import Spec, checkCompatability
from fabrics.diffGeometry.diffMap import DifferentialMap, RelativeDifferentialMap

from fabrics.helpers.functions import joinRefTrajs
from fabrics.helpers.variables import Variables
from fabrics.helpers.casadiFunctionWrapper import CasadiFunctionWrapper


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
        assert isinstance(l, ca.SX)
        if 'x' in kwargs:
            self._vars = Variables(state_variables={"x": kwargs.get('x'), "xdot": kwargs.get('xdot')})
        elif 'var' in kwargs:
            self._vars = kwargs.get('var')
        self._rel = False
        self._refTrajs = []
        if 'refTrajs' in kwargs:
            self._refTrajs = kwargs.get('refTrajs')
            self._rel = len(self._refTrajs) > 0
        self.applyEulerLagrange()

    def x(self):
        return self._vars.position_variable()

    def xdot(self):
        return self._vars.velocity_variable()

    def xdot_rel(self):
        if self._rel:
            return self.xdot() - self._refTrajs[-1].xdot()
        else:
            return self.xdot()

    def __add__(self, b):
        assert isinstance(b, Lagrangian)
        checkCompatability(self, b)
        refTrajs = joinRefTrajs(self._refTrajs, b._refTrajs)
        return Lagrangian(self._l + b._l, var=self._vars, refTrajs=refTrajs)

    def applyEulerLagrange(self):
        dL_dxdot = ca.gradient(self._l, self.xdot())
        dL_dx = ca.gradient(self._l, self.x())
        d2L_dxdxdot = ca.jacobian(dL_dx, self.xdot())
        d2L_dxdot2 = ca.jacobian(dL_dxdot, self.xdot())
        f_rel = np.zeros(self.x().size()[0])
        en_rel = np.zeros(1)

        for refTraj in self._refTrajs:
            x_ref, xdot_ref, xddot_ref = refTraj.parameters()
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


    def evaluate(self, **kwargs):
        funs = self._funs.evaluate(**kwargs)
        H = funs['H']
        M, f, _ = self._S.evaluate(**kwargs)
        return M, f, H

    def pull(self, dm: DifferentialMap):
        assert isinstance(dm, DifferentialMap)
        l_subst = ca.substitute(self._l, self.x(), dm._phi)
        l_subst2 = ca.substitute(l_subst, self.xdot(), dm.phidot())
        new_vars = dm._vars
        if hasattr(dm, '_refTraj'):
            refTrajs = [dm._refTraj] + [refTraj.pull(dm) for refTraj in self._refTrajs]
        else:
            refTrajs = [refTraj.pull(dm) for refTraj in self._refTrajs]
        return Lagrangian(l_subst2, var=new_vars, refTrajs=refTrajs)


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
