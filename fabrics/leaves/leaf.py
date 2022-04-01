import casadi as ca

class Leaf(object):
    def __init__(self, q: ca.SX, qdot: ca.SX):
        self._q = q
        self._qdot = qdot
        self._p = {}
        self._dm = None
        self._lag = None
        self._geo = None

    def set_params(self, **kwargs):
        for key in self._p.keys():
            if key in kwargs:
                self._p[key] = kwargs.get(key)

    def geometry(self):
        return self._geo

    def map(self):
        return self._dm

    def lagrangian(self):
        return self._lag
