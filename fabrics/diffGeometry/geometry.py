import casadi as ca
import numpy as np
from copy import deepcopy
import logging

from fabrics.diffGeometry.diffMap import DifferentialMap, DynamicDifferentialMap
from fabrics.helpers.functions import joinVariables
from fabrics.helpers.variables import Variables
from fabrics.helpers.casadiFunctionWrapper import CasadiFunctionWrapper


class Geometry:
    """description"""

    def __init__(self, **kwargs):
        self._x_ref_name = "x_ref"
        self._xdot_ref_name = "xdot_ref"
        self._xddot_ref_name = "xddot_ref"
        if 'ref_names' in kwargs:
            ref_names = kwargs.get('ref_names')
            self._x_ref_name = ref_names[0]
            self._xdot_ref_name = ref_names[1]
            self._xddot_ref_name = ref_names[2]
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

    def is_dynamic(self) -> bool:
        logging.debug(f"{self._x_ref_name}, {self._vars.parameters()}")
        return self._x_ref_name in self._vars.parameters()

    def __add__(self, b):
        assert isinstance(b, Geometry)
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
        if b.is_dynamic():
            ref_names += b.ref_names()
        if len(ref_names) > 0:
            ref_arguments = {'ref_names': ref_names}
        else:
            ref_arguments = {}
        return Geometry(h=self._h + b._h, var=all_vars, *ref_arguments)

    def ref_names(self) -> list:
        return [self._x_ref_name, self._xdot_ref_name, self._xddot_ref_name]

    def pull(self, dm: DifferentialMap):
        assert isinstance(dm, DifferentialMap)
        h_pulled = ca.mtimes(ca.pinv(dm._J), self._h + dm.Jdotqdot())
        h_pulled_subst_x = ca.substitute(h_pulled, self.x(), dm._phi)
        h_pulled_subst_x_xdot = ca.substitute(h_pulled_subst_x, self.xdot(), dm.phidot())
        new_state_variables = dm.state_variables()
        new_parameters = {}
        new_parameters.update(self._vars.parameters())
        new_parameters.update(dm.params())
        new_vars = Variables(state_variables=new_state_variables, parameters=new_parameters)
        if hasattr(dm, '_refTraj'):
            refTrajs = [dm._refTraj] + [refTraj.pull(dm) for refTraj in self._refTrajs]
        else:
            refTrajs = [refTraj.pull(dm) for refTraj in self._refTrajs]
        return Geometry(h=h_pulled_subst_x_xdot, var=new_vars, refTrajs=refTrajs)

    def dynamic_pull(self, dm: DynamicDifferentialMap):
        h_pulled = self._h - dm.xddot_ref()
        h_pulled_subst_x = ca.substitute(h_pulled, self.x(), dm._phi)
        h_pulled_subst_x_xdot = ca.substitute(h_pulled_subst_x, self.xdot(), dm.phidot())
        return Geometry(h=h_pulled_subst_x_xdot, var=dm._vars)

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
