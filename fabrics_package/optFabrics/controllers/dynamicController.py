import casadi as ca
import numpy as np

# fabrics
from optFabrics.creators.attractors import *
from optFabrics.creators.repellers import *
from optFabrics.rootGeometry import RootGeometry
from optFabrics.damper import createRootDamper, createConstantRootDamper
from optFabrics.diffMap import DiffMap


class DynamicController(object):
    def __init__(self, n, q_ca, qdot_ca):
        self._n = n
        self._q_ca = q_ca
        self._qdot_ca = qdot_ca
        self._leaves = []
        self._m_forcing = 0

    def addAttractor(self, xd_t, t_ca, m, fk,k=5.0):
        x = ca.SX.sym("x", m)
        xdot = ca.SX.sym("xdot", m)
        xd_ca = ca.SX.sym("xd", m)
        self._m_forcing += m
        attractor = createDynamicAttractor(self._q_ca, self._qdot_ca, x, xdot, xd_ca, xd_t, t_ca, fk,k=k)
        self._leaves.append(attractor)

    def addStaticAttractor(self, xd, m, fk, k=5):
        x = ca.SX.sym("x", m)
        xdot = ca.SX.sym("xdot", m)
        self._m_forcing += m
        attractor = createAttractor(self._q_ca, self._qdot_ca, x, xdot, xd, fk, k=k)
        self._leaves.append(attractor)

    def addSplineAttractor(self, spline, m, fk, T, dt, k=5):
        x = ca.SX.sym("x", m)
        xdot = ca.SX.sym("xdot", m)
        self._m_forcing += m
        attractor = createSplineAttractor(self._q_ca, self._qdot_ca, x, xdot, spline, T, dt, fk, k=k)
        self._leaves.append(attractor)

    def addDiffDriveAttractor(self, spline, m, fk, T, k=5):
        x = ca.SX.sym("x", m)
        xdot = ca.SX.sym("xdot", m)
        self._m_forcing = m
        attractor = createDiffDriveSplineAttractor(self._q_ca, self._qdot_ca, x, xdot, spline, T, fk, k=k)
        self._leaves.append(attractor)

    def addRedundancyRes(self):
        q0 = np.zeros(self._n)
        attractor = createRedundancySolver(self._q_ca, self._qdot_ca, q0)
        self._leaves.append(attractor)

    def addDamper(self, m, fk, b=np.array([0.03, 6.5])):
        x_ex = ca.SX.sym("x_ex", m)
        xdot_ex = ca.SX.sym("xdot_ex", m)
        phi_ex = fk
        diffMap_ex = DiffMap(
            "exec_map", phi_ex, self._q_ca, self._qdot_ca, x_ex, xdot_ex
        )
        self._rootDamper = createRootDamper(
            self._q_ca, self._qdot_ca, self._m_forcing, diffMap_ex, x_ex, xdot_ex, b=b
        )

    def addConstantDamper(self):
        self._rootDamper = createConstantRootDamper(beta=3.0)

    def assembleRootGeometry(self, m=1):
        le_root = m * 1.0 / 2.0 * ca.dot(self._qdot_ca, self._qdot_ca)
        self._rootGeo = RootGeometry(
            self._leaves, le_root, self._n, self._q_ca, self._qdot_ca, damper=self._rootDamper
        )

    def addJointLimits(self, lower_lim, upper_lim):
        limit_leaves =  createJointLimits(self._q_ca, self._qdot_ca, upper_lim, lower_lim)
        self._leaves += limit_leaves

    def addObstacles(self, obsts, fk):
        lcols = []
        for obst in obsts:
            x_obst = obst.x()
            r_obst = obst.r()
            lcols.append(
                createCollisionAvoidance(self._q_ca, self._qdot_ca, fk, x_obst, r_obst)
            )
        self._leaves += lcols

    def addPlane(self, plane, fk):
        lplane = createPlaneAvoidance(self._q_ca, self._qdot_ca, fk, plane)
        self._leaves.append(lplane)

    def computeAction(self, z, t):
        zdot = self._rootGeo.contDynamics(z, t)
        u = zdot[self._n : 2 * self._n]
        return u

