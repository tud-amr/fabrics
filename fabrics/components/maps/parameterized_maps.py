import casadi as ca
from fabrics.diffGeometry.diffMap import (
    ParameterizedDifferentialMap,
)
from fabrics.helpers.variables import Variables


class ParameterizedGoalMap(ParameterizedDifferentialMap):
    def __init__(self, var, fk, reference_variable):
        phi = fk - reference_variable
        super().__init__(phi, var=var)

class ParameterizedGeometryMap(ParameterizedDifferentialMap):
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
        super().__init__(phi, var=var)
