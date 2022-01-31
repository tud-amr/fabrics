import casadi as ca
from fabrics.diffGeometry.diffMap import DifferentialMap


class CollisionMap(DifferentialMap):
    def __init__(self, q, qdot, fk, x_obst, r_obst, r_body=0.0):
        phi = ca.norm_2(fk - x_obst) / (r_obst + r_body) - 1
        super().__init__(phi, q=q, qdot=qdot)

class SelfCollisionMap(DifferentialMap):
    def __init__(self, q, qdot, fk1, fk2, r_body=0.0):
        phi = ca.norm_2(fk1 - fk2) / (2 * r_body) - 1
        super().__init__(phi, q=q, qdot=qdot)

class UpperLimitMap(DifferentialMap):
    def __init__(self, q, qdot, limit, index):
        phi = limit - q[index]
        super().__init__(phi, q=q, qdot=qdot)

class LowerLimitMap(DifferentialMap):
    def __init__(self, q, qdot, limit, index):
        phi = q[index] - limit
        super().__init__(phi, q=q, qdot=qdot)


class GoalMap(DifferentialMap):
    def __init__(self, q, qdot, fk, goal):
        phi = fk - goal
        super().__init__(phi, q=q, qdot=qdot)

