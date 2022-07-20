import casadi as ca
import re
import numpy as np

from fabrics.helpers.exceptions import SpecException
from fabrics.helpers.exceptions import ExpressionSparseError


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

def is_sparse(expression: ca.SX) -> bool:
    return not ca.symvar(expression)

def symbolic(name: str):
    return ca.SX.sym(name, 1)

def sym(name: str):
    return symbolic(name)

def parse_symbolic_input(expression: str, x: ca.SX, xdot: ca.SX, name: str = '') -> tuple:
    if len(name) > 0:
        name = '_' + name
    new_parameters = {}
    parameters = re.findall(r"\(\'(\w*)\'\)", expression)
    for i, _ in enumerate(parameters):
        expression = expression.replace(parameters[i], parameters[i] + name)
        parameters[i] += name

    symbolic_expression = eval(expression)
    if isinstance(symbolic_expression, ca.SX):
        all_variables = ca.symvar(symbolic_expression)
    else:
        all_variables = []
    for variable in all_variables:
        if variable.name() in parameters:
            new_parameters[variable.name()] = variable
    return new_parameters, symbolic_expression


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


def joinRefTrajs(refTrajs1, refTrajs2):
    refTrajs = refTrajs1 + refTrajs2
    unique_items = []
    for item in refTrajs:
        already_exists = False
        for u_item in unique_items:
            parameter_list_1 = list(u_item._vars.parameters().values())
            parameter_list_2 = list(item._vars.parameters().values())
            if u_item == item or ca.is_equal(parameter_list_1[0], parameter_list_2[0]):
                already_exists = True
                break
        if not already_exists:
            unique_items.append(item)
    return unique_items
