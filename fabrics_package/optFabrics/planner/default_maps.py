import casadi as ca
import numpy as np
from optFabrics.diffGeometry.diffMap import DifferentialMap, VariableDifferentialMap

class CollisionMap(DifferentialMap):
    def __init__(self, q, qdot, fk, x_obst, r_obst):
        phi = ca.norm_2(fk - x_obst) / r_obst - 1
        super().__init__(phi, q=q, qdot=qdot)


class VariableCollisionMap(VariableDifferentialMap):
    def __init__(self, q, qdot, fk, r_obst, q_p, qdot_p, qddot_p):
        phi = ca.norm_2(fk - q_p) / r_obst - 1
        super().__init__(phi, q=q, qdot=qdot, q_p=q_p, qdot_p=qdot_p, qddot_p=qddot_p)


class GoalMap(DifferentialMap):
    def __init__(self, q, qdot, fk, goal):
        phi = fk - goal
        super().__init__(phi, q=q, qdot=qdot)


class VariableGoalMap(VariableDifferentialMap):
    def __init__(self, q, qdot, fk, q_g, qdot_g, qddot_g):
        phi = fk - q_g
        super().__init__(phi, q=q, qdot=qdot, q_p=q_g, qdot_p=qdot_g, qddot_p=qddot_g)
