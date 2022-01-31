import casadi as ca
from fabrics.diffGeometry.geometry import Geometry
from fabrics.diffGeometry.variables import eps


class CollisionGeometry(Geometry):
    def __init__(self, x, xdot, **kwargs):
        p = {"lam": 2, 'exp': 1}
        for key in p.keys():
            if key in kwargs:
                p[key] = kwargs.get(key)
        s = -0.5 * (ca.sign(xdot) - 1)
        h = -p["lam"] / (x ** p["exp"]) * s * xdot ** 2
        super().__init__(h=h, x=x, xdot=xdot)

class LimitGeometry(Geometry):
    def __init__(self, x, xdot, **kwargs):
        p = {'lam': 0.25, 'a1': 0.4, 'a2': 0.2, 'a3': 20, 'a4': 5, 'exp': 1}
        for key in p.keys():
            if key in kwargs:
                p[key] = kwargs.get(key)
        # psi = p['a1']/(x**2) + p['a2'] * ca.log(ca.exp(-p['a3'] * (x - p['a4'])) + 1)
        # h = p['lam'] * xdot**2 * ca.gradient(psi, x)
        s = -0.5 * (ca.sign(xdot) - 1)
        h = -p["lam"] / (x ** p["exp"]) * s * xdot ** 2
        super().__init__(h=h, x=x, xdot=xdot)


class GoalGeometry(Geometry):
    def __init__(self, x, xdot, **kwargs):
        p = {"k_psi": 10, "a_psi": 10}
        for key in p.keys():
            if key in kwargs:
                p[key] = kwargs.get(key)
        psi = p["k_psi"] * (
            ca.norm_2(x)
            + 1 / p["a_psi"] * ca.log(1 + ca.exp(-2 * p["a_psi"] * ca.norm_2(x)))
        )
        h_psi = ca.gradient(psi, x)
        super().__init__(h=h_psi, x=x, xdot=xdot)
