from fabrics.helpers.variables import Variables

class Leaf(object):
    def __init__(self, var: Variables):
        self._var_q = var
        self._p = {}
        self._dm = None
        self._lag = None
        self._geo = None

    def set_params(self, **kwargs):
        for key in self._p:
            if key in kwargs:
                self._p[key] = kwargs.get(key)

    def geometry(self):
        return self._geo

    def map(self):
        return self._dm

    def lagrangian(self):
        return self._lag
