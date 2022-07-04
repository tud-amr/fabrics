import casadi as ca
from fabrics.diffGeometry.diffMap import DifferentialMap
from fabrics.helpers.variables import Variables


class CollisionMap(DifferentialMap):
    def __init__(self, var, fk, x_obst, r_obst, r_body=0.0):
        phi = ca.norm_2(fk - x_obst) / (r_obst + r_body) - 1
        super().__init__(phi, var=var)

class SelfCollisionMap(DifferentialMap):
    def __init__(self, var_q, fk1, fk2, r_body=0.0):
        phi = ca.norm_2(fk1 - fk2) / (2 * r_body) - 1
        super().__init__(phi, var=var_q)

class UpperLimitMap(DifferentialMap):
    def __init__(self, var_q, limit, index):
        q = var_q.position_variable()
        phi = limit - q[index]
        super().__init__(phi, var=var_q)

class LowerLimitMap(DifferentialMap):
    def __init__(self, var_q, limit, index):
        q = var_q.position_variable()
        phi = q[index] - limit
        super().__init__(phi, var=var_q)


class GoalMap(DifferentialMap):
    def __init__(self, var_q, fk, goal):
        phi = fk - goal
        super().__init__(phi, var=var_q)

class ParameterizedGoalMap(DifferentialMap):
    def __init__(self, var, fk, reference_variable):
        phi = fk - reference_variable
        super().__init__(phi, var=var)

class ParameterizedGeometryMap(DifferentialMap):
    def __init__(self, var, fk):
        x_geo = list(var.parameters().values())[-1]
        r_obst = 0.5
        r_body = 0.5
        phi = ca.norm_2(fk - x_geo) / (r_obst + r_body) - 1
        super().__init__(phi, var=var)

class ParameterizedGeometryMap(DifferentialMap):
    pass

class ParameterizedObstacleMap(ParameterizedGeometryMap):
    def __init__(self, var: Variables, fk, reference_variable, radius_variable, radius_body_variable):
        phi = ca.norm_2(fk - reference_variable) / (radius_variable + radius_body_variable) - 1
        super().__init__(phi, var=var)

