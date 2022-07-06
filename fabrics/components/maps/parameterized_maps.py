import casadi as ca
from fabrics.diffGeometry.diffMap import (
    DifferentialMap,
)
from fabrics.helpers.variables import Variables


class ParameterizedGoalMap(DifferentialMap):
    def __init__(self, var, fk, reference_variable):
        phi = fk - reference_variable
        super().__init__(phi, var)

class ParameterizedGeometryMap(DifferentialMap):
    pass

class ParameterizedObstacleMap(ParameterizedGeometryMap):
    def __init__(
        self,
        var: Variables,
        fk,
        reference_variable,
        radius_variable,
        radius_body_variable,
    ):
        phi = (
            ca.norm_2(fk - reference_variable)
            / (radius_variable + radius_body_variable)
            - 1
        )
        super().__init__(phi, var)


"""
class ParameterizedObstacleMap(ParameterizedGeometryMap):
    def __init__(
        self,
        var: Variables,
        fk,
        reference_variable,
        radius_variable,
        radius_body_variable,
    ):
        phi = (
            ca.norm_2(fk - reference_variable)
            / (radius_variable + radius_body_variable)
            - 1
        )
        super().__init__(phi, var=var)
"""
