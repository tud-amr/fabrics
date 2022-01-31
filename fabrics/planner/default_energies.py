import casadi as ca
import numpy as np
from fabrics.diffGeometry.energy import Lagrangian


class CollisionLagrangian(Lagrangian):
    def __init__(self, x, xdot, **kwargs):
        p = {'lam': 2.0}
        for key in p.keys():
            if key in kwargs:
                p[key] = kwargs.get(key)
        # s cannot be used here as it contradict positivity of the energy
        # s = -0.5 * (ca.sign(xdot) - 1)
        # le = p['lam'] * 1/(x**2) * s * xdot**2
        le = p['lam'] * 1/(x**2) * xdot**2
        super().__init__(le, x=x, xdot=xdot)


class GoalLagrangian(Lagrangian):
    def __init__(self, x, xdot, **kwargs):
        p = {'m': [0.3, 2.0], 'a_m': 0.75}
        for key in p.keys():
            if key in kwargs:
                p[key] = kwargs.get(key)
        M_psi = ((p['m'][1] - p['m'][0]) * ca.exp(-1*(p['a_m'] * ca.norm_2(x))**2) + p['m'][0]) * ca.SX(np.identity(x.size()[0]))
        le_psi = ca.dot(xdot, ca.mtimes(M_psi, xdot))
        super().__init__(le_psi, x=x, xdot=xdot)


class ExecutionLagrangian(Lagrangian):
    def __init__(self, x, xdot):
        le = 0.5 * ca.dot(xdot, xdot)
        super().__init__(le, x=x, xdot=xdot)
