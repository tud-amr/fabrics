import casadi as ca
from optFabrics.diffGeometry.geometry import Geometry

class CollisionGeometry(Geometry):
    def __init__(self, x, xdot, **kwargs):
        p = {'lam' : 2}
        for key in p.keys():
            if key in kwargs:
                p[key] = kwargs.get(key)
        h = -p['lam'] / (x ** 1) * xdot**2
        super().__init__(h=h, x=x, xdot=xdot)


class GoalGeometry(Geometry):
    def __init__(self, x, xdot, **kwargs):
        p = {'k_psi' : 10, 'a_psi': 10}
        for key in p.keys():
            if key in kwargs:
                p[key] = kwargs.get(key)
        psi = p['k_psi'] * (ca.norm_2(x) + 1/p['a_psi'] * ca.log(1 + ca.exp(-2 * p['a_psi'] * ca.norm_2(x))))
        h_psi = ca.gradient(psi, x)
        super().__init__(h=h_psi, x=x, xdot=xdot)


