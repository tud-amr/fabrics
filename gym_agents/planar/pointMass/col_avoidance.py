import gym
import planarenvs.pointRobot
import casadi as ca
import numpy as np

from fabrics.planner.fabricPlanner import FabricPlanner
from fabrics.diffGeometry.diffMap import DifferentialMap
from fabrics.diffGeometry.energy import Lagrangian
from fabrics.diffGeometry.geometry import Geometry

from MotionPlanningEnv.sphereObstacle import SphereObstacle

def pointMassAvoidance(n_steps=1200, render=True):
    ## setting up the problem
    obstDict = {'dim': 2, 'type': 'sphere', 'geometry': {'position': [0.0, 0.0], 'radius': 1.0}} 
    obst = SphereObstacle(name="obst", contentDict=obstDict)
    n = 2
    q = ca.SX.sym("q", n)
    qdot = ca.SX.sym("qdot", n)
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    l_base = 1.0 * ca.dot(qdot, qdot)
    h_base = ca.SX(np.zeros(n))
    baseGeo = Geometry(h=h_base, x=q, xdot=qdot)
    baseLag = Lagrangian(l_base, x=q, xdot=qdot)
    planner = FabricPlanner(baseGeo, baseLag)
    phi = ca.norm_2(q - obst.position()) / obst.radius() - 1
    dm = DifferentialMap(phi, q=q, qdot=qdot)
    s = -0.5 * (ca.sign(xdot) - 1)
    lam = 5.00
    le = lam * 1/x * s * xdot**2
    lag_col = Lagrangian(le, x=x, xdot=xdot)
    h = -lam / (x ** 3) * xdot**2
    geo = Geometry(h=h, x=x, xdot=xdot)
    planner.addGeometry(dm, lag_col, geo)
    l_ex = 0.5 * ca.dot(qdot, qdot)
    exLag = Lagrangian(l_ex, x=q, xdot=qdot)
    exLag.concretize()
    planner.setExecutionEnergy(exLag)
    planner.concretize()
    # setup environment
    x0 = np.array([2.3, 0.5])
    xdot0 = np.array([-1.0, -0.0])
    # running the simulation
    env = gym.make('point-robot-acc-v0', dt=0.01, render=render)
    ob = env.reset(pos=x0, vel=xdot0)
    env.addObstacle(obst)
    print("Starting episode")
    for i in range(n_steps):
        if i % 100 == 0:
            print('time step : ', i)
        # t0 = time.time()
        action = planner.computeAction(ob['x'], ob['xdot'])
        _, _, en_ex = exLag.evaluate(ob['x'], ob['xdot'])
        print(f"Execution Energy : {en_ex}")
        ob, reward, done, info = env.step(action)
    return {}


if __name__ == "__main__":
    n_steps = 1200
    pointMassAvoidance(n_steps)
