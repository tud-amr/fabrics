import casadi as ca
import numpy as np

# fabrics
from optFabrics.creators.attractors import createAttractor, createRedundancySolver
from optFabrics.creators.repellers import createCollisionAvoidance
from optFabrics.rootGeometry import RootGeometry
from optFabrics.damper import createRootDamper
from optFabrics.diffMap import DiffMap


class StaticController(object):
    def __init__(self, n, q_ca, qdot_ca):
        self._n = n
        self._q_ca = q_ca
        self._qdot_ca = qdot_ca
        self._leaves = []

    def addAttractor(self, xd, m, fk):
        x = ca.SX.sym("x", m)
        xdot = ca.SX.sym("xdot", m)
        self._m_forcing = m
        attractor = createAttractor(self._q_ca, self._qdot_ca, x, xdot, xd, fk)
        self._leaves.append(attractor)

    def addRedundancyRes(self):
        q0 = np.zeros(self._n)
        attractor = createRedundancySolver(self._q_ca, self._qdot_ca, q0)
        self._leaves.append(attractor)

    def addDamper(self, m, fk):
        x_ex = ca.SX.sym("x_ex", m)
        xdot_ex = ca.SX.sym("xdot_ex", m)
        phi_ex = fk
        diffMap_ex = DiffMap(
            "exec_map", phi_ex, self._q_ca, self._qdot_ca, x_ex, xdot_ex
        )
        self._rootDamper = createRootDamper(
            self._q_ca, self._qdot_ca, self._m_forcing, diffMap_ex, x_ex, xdot_ex
        )

    def assembleRootGeometry(self):
        le_root = 1.0 / 2.0 * ca.dot(self._qdot_ca, self._qdot_ca)
        self._rootGeo = RootGeometry(
            self._leaves, le_root, self._n, damper=self._rootDamper
        )

    def addObstacles(self, obsts, fk):
        lcols = []
        for obst in obsts:
            x_obst = obst.x()
            r_obst = obst.r()
            lcols.append(
                createCollisionAvoidance(self._q_ca, self._qdot_ca, fk, x_obst, r_obst)
            )
        self._leaves += lcols

    def computeAction(self, z, t):
        zdot = self._rootGeo.contDynamics(z, t)
        u = zdot[self._n : 2 * self._n]
        return u
