import casadi as ca
from fabrics.diffGeometry.energy import Lagrangian

class ExecutionLagrangian(Lagrangian):
    def __init__(self, var):
        xdot = var.velocity_variable()
        le = 0.5 * ca.dot(xdot, xdot)
        super().__init__(le, var=var)
