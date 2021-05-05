import casadi as ca
import numpy as np
# gym
import gym
import pointRobot
# fabrics
from optFabrics.creators.attractors import createAttractor, createSplineAttractor
from optFabrics.creators.repellers import createCollisionAvoidance
from optFabrics.rootGeometry import RootGeometry
from optFabrics.damper import createRootDamper
from optFabrics.diffMap import DiffMap
#
from obstacle import Obstacle
from robotPlot import RobotPlot

from geomdl import BSpline
from geomdl import utilities
from geomdl.visualization import VisMPL

class DynamicFabricController():
    def __init__(self, spline, obsts, T, x_d, static=False):
        self._n = 2
        q = ca.SX.sym("q", 2)
        qdot = ca.SX.sym("qdot", 2)
        x = ca.SX.sym("x", 2)
        xdot = ca.SX.sym("xdot", 2)
        fk = q
        ls = []
        lcols = []
        fk_col = q
        for obst in obsts:
            x_obst = obst.x()
            r_obst = obst.r()
            lcols.append(createCollisionAvoidance(q, qdot, fk_col, x_obst, r_obst))
        ls += lcols
        lforcing_dyn = createSplineAttractor(q, qdot, x, xdot, spline, T, fk)
        ls.append(lforcing_dyn)
        if static==True:
            lforcing_sta = createAttractor(q, qdot, x, xdot, x_d, fk, k=20.0)
            ls.append(lforcing_sta)
        x_ex = ca.SX.sym("x_ex", 2)
        xdot_ex = ca.SX.sym("xdot_ex", 2)
        phi_ex = q
        diffMap_ex = DiffMap("exec_map", phi_ex, q, qdot, x_ex, xdot_ex)
        rootDamper = createRootDamper(q, qdot, 2, diffMap_ex, x_ex, xdot_ex)
        le_root = 1.0/2.0 * ca.dot(qdot, qdot)
        self._rg_forced = RootGeometry(ls, le_root, 2, damper=rootDamper)

    def computeAction(self, z, t):
        zdot = self._rg_forced.contDynamics(z, t)
        u = zdot[self._n:2 * self._n]
        return u


def main():
    ## setting up the problem
    t_ca = ca.SX.sym("t", 1)
    x_d = np.array([-2.0, -1.0])
    x0 = np.array([3.0, 2.0])
    xdot0 = np.array([0.0, 0.0])
    obsts = [
        Obstacle(np.array([0.5, 0.0]), 0.8), 
        Obstacle(np.array([-1.5, 1.0]), 0.2), 
        Obstacle(np.array([0.0, 1.0]), 0.8)
        ]
    T = 5.0
    con = FabricController(x_d, obsts)
    crv = BSpline.Curve()
    crv.degree = 2
    crv.ctrlpts = [x0.tolist(), [2, -1.0], x_d.tolist()]
    crv.ctrlpts = [x0.tolist(), [-1.0, 3.0], x_d.tolist()]
    #crv.ctrlpts = [[1, 0], [2, 1], [3, 2]]
    crv.knotvector = utilities.generate_knot_vector(crv.degree, len(crv.ctrlpts))
    con_dynamic = DynamicFabricController(crv, obsts, T, x_d, static=False)
    con_dynamic2 = DynamicFabricController(crv, obsts, T, x_d, static=True)
    cons = [con, con_dynamic, con_dynamic2]
    dt = 0.005
    n_steps = int(T/dt)
    qs = []
    env = gym.make('point-robot-acc-v0', dt = dt)
    for con in cons:
        t = 0.0
        ob = env.reset(x0, xdot0)
        q = np.zeros((n_steps, 2))
        for i in range(n_steps):
            if i % 100 == 0:
                print('time step : ', i)
            action = np.array([0.0, 0.2])
            action = con.computeAction(ob, t)
            t += dt
            #env.render()
            ob, reward, done, info = env.step(action)
            q[i, :] = ob[0:2]
        qs.append(q)

    fk_fun = lambda q : q
    robotPlot = RobotPlot(qs, fk_fun, 2, types=[0, 0, 0])
    robotPlot.initFig(2, 2)
    robotPlot.plot()
    robotPlot.addObstacle([0, 1, 2], obsts)
    robotPlot.makeAnimation(n_steps)
    robotPlot.show()

if __name__ == "__main__":
    main()
