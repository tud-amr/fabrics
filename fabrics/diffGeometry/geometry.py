import casadi as ca
import numpy as np
from copy import deepcopy

from fabrics.diffGeometry.diffMap import DifferentialMap
from fabrics.helpers.constants import eps
from fabrics.helpers.functions import joinVariables
from fabrics.helpers.variables import Variables
from fabrics.helpers.casadiFunctionWrapper import CasadiFunctionWrapper


class Geometry:
    """description"""

    def __init__(self, **kwargs):
        if 'x' in kwargs:
            h = kwargs.get("h")
            self._vars = Variables(state_variables={"x": kwargs.get('x'), "xdot": kwargs.get('xdot')})
        elif 'var' in kwargs:
            h = kwargs.get("h")
            self._vars = kwargs.get('var')
        elif 's' in kwargs:
            s = kwargs.get("s")
            h = s.h()
            self._vars = s._vars
        self._refTrajs = []
        if 'refTrajs' in kwargs:
            self._refTrajs = kwargs.get('refTrajs')

        assert isinstance(h, ca.SX)
        self._h = h

    def x(self):
        return self._vars.position_variable()

    def xdot(self):
        return self._vars.velocity_variable()

    def __add__(self, b):
        assert isinstance(b, Geometry)
        # TODO: checkCompatibility()  <24-07-21, mspahn> #
        nb_vars_a = len(self._vars)
        nb_vars_b = len(b._vars)
        if nb_vars_a >= nb_vars_b:
            var = self._vars
        else:
            var = b._vars
        return Geometry(h=self._h + b._h, var=var)

    def pull(self, dm: DifferentialMap):
        assert isinstance(dm, DifferentialMap)
        Jt = ca.transpose(dm._J)
        JtJ = ca.mtimes(Jt, dm._J)
        h_1 = ca.mtimes(Jt, self._h)
        h_2 = ca.mtimes(Jt, dm.Jdotqdot())
        JtJ_eps = JtJ + np.identity(dm.q().size()[0]) * eps
        h_pulled = ca.mtimes(ca.pinv(JtJ_eps), h_1 + h_2)
        h_pulled_subst_x = ca.substitute(h_pulled, self.x(), dm._phi)
        h_pulled_subst_x_xdot = ca.substitute(h_pulled_subst_x, self.xdot(), dm.phidot())
        new_vars = dm._vars
        if hasattr(dm, '_refTraj'):
            refTrajs = [dm._refTraj] + [refTraj.pull(dm) for refTraj in self._refTrajs]
        else:
            refTrajs = [refTraj.pull(dm) for refTraj in self._refTrajs]
        return Geometry(h=h_pulled_subst_x_xdot, var=new_vars, refTrajs=refTrajs)

    def concretize(self):
        self._xddot = -self._h
        var = deepcopy(self._vars)
        for refTraj in self._refTrajs:
            var += refTraj._vars
        self._funs = CasadiFunctionWrapper(
            "funs", var.asDict(), {"h": self._h, "xddot": self._xddot}
        )

    def evaluate(self, **kwargs):
        evaluations = self._funs.evaluate(**kwargs)
        h_eval = evaluations['h']
        xddot_eval = evaluations['xddot']
        return [h_eval, xddot_eval]

    def testHomogeneousDegree2(self):
        x = np.random.rand(self.x().size()[0])
        xdot = np.random.rand(self.xdot().size()[0])
        alpha = 2.0
        xdot2 = alpha * xdot
        h, _ = self.evaluate(x=x, xdot=xdot)
        h2, _ = self.evaluate(x=x, xdot=xdot2)
        return h * alpha ** 2 == h2
