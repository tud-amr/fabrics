import casadi as ca
from fabrics.diffGeometry.diffMap import DifferentialMap, ParameterizedDifferentialMap


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

class ParameterizedGoalMap(ParameterizedDifferentialMap):
    def __init__(self, q, qdot, fk, goal):
        phi = fk - goal
        super().__init__(phi, goal, q=q, qdot=qdot)

