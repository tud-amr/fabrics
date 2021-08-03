import casadi as ca
import numpy as np
import inspect

from optFabrics.diffGeometry.diffMap import DifferentialMap
from optFabrics.diffGeometry.variables import eps


class SpecException(Exception):
    def __init__(self, expression, message):
        self._expression = expression
        self._message = message

    def what(self):
        return self._expression + ": " + self._message


def checkCompatability(a, b):
    if a.x().size() != b.x().size():
        raise SpecException(
            "Operation invalid",
            "Different dimensions: " + str(a.x().size()) + " vs. " + str(b.x().size()),
        )
    if not (ca.is_equal(a.x(), b.x())):
        raise SpecException(
            "Operation invalid",
            "Different variables: " + str(a.x()) + " vs. " + str(b.x()),
        )


def joinVariables(var1, var2):
    var = var1 + var2
    unique_items = []
    for item in var:
        already_exists = False
        for u_item in unique_items:
            if u_item.size() == item.size() and ca.is_equal(u_item, item):
                already_exists = True
                break
        if not already_exists:
            unique_items.append(item)
    return unique_items



class Spec:
    """description"""

    def __init__(self, M: ca.SX, f: ca.SX, **kwargs):
        if len(kwargs) == 2:
            x = kwargs.get('x')
            xdot = kwargs.get('xdot')
            self._vars = [x, xdot]
        elif len(kwargs) == 1:
            self._vars = kwargs.get('var')
        for var in self._vars:
            assert isinstance(var, ca.SX)
        assert isinstance(M, ca.SX)
        assert isinstance(f, ca.SX)
        if self.x().size() != self.xdot().size():
            raise SpecException(
                "Attempted spec creation failed",
                "Different dimensions of x : "
                + str(b.x().size())
                + " and xdot :"
                + str(self.x().size()),
            )
        self._M = M
        self._f = f

    def x(self):
        return self._vars[0]

    def xdot(self):
        return self._vars[1]

    def concretize(self):
        self._xddot = ca.mtimes(
            ca.pinv(self._M + np.identity(self.x().size()[0]) * eps), -self._f
        )
        self._funs = ca.Function(
            "M", self._vars, [self._M, self._f, self._xddot]
        )

    def evaluate(self, *args):
        for arg in args:
            assert isinstance(arg, np.ndarray)
        funs = self._funs(*args)
        M_eval = np.array(funs[0])
        f_eval = np.array(funs[1])[:, 0]
        xddot_eval = np.array(funs[2])[:, 0]
        return [M_eval, f_eval, xddot_eval]

    def __add__(self, b):
        assert isinstance(b, Spec)
        checkCompatability(self, b)
        all_vars = joinVariables(self._vars, b._vars)
        return Spec(self._M + b._M, self._f + b._f, var=all_vars)

    def pull(self, dm: DifferentialMap):
        assert isinstance(dm, DifferentialMap)
        M_pulled = ca.mtimes(ca.transpose(dm._J), ca.mtimes(self._M, dm._J))
        f_1 = ca.mtimes(
            ca.transpose(dm._J), ca.mtimes(self._M, dm.Jdotqdot())
        )
        f_2 = ca.mtimes(ca.transpose(dm._J), self._f)
        f_pulled = f_1 + f_2
        M_pulled_subst_x = ca.substitute(M_pulled, self.x(), dm._phi)
        M_pulled_subst_x_xdot = ca.substitute(
            M_pulled_subst_x, self.xdot(), ca.mtimes(dm._J, dm.qdot())
        )
        f_pulled_subst_x = ca.substitute(f_pulled, self.x(), dm._phi)
        f_pulled_subst_x_xdot = ca.substitute(
            f_pulled_subst_x, self.xdot(), dm.phidot()
        )
        return Spec(M_pulled_subst_x_xdot, f_pulled_subst_x_xdot, var=dm._vars)
