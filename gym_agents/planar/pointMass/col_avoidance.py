import gym
import planarenvs.point_robot
import casadi as ca
import numpy as np

from fabrics.planner.fabricPlanner import FabricPlanner
from fabrics.diffGeometry.diffMap import DifferentialMap
from fabrics.diffGeometry.energy import Lagrangian
from fabrics.diffGeometry.geometry import Geometry

from fabrics.helpers.variables import Variables

from MotionPlanningEnv.sphereObstacle import SphereObstacle

def pointMassAvoidance(n_steps=1200, render=True):
    ## setting up the problem
    obstDict = {'dim': 2, 'type': 'sphere', 'geometry': {'position': [0.0, 0.0], 'radius': 1.0}} 
    obst = SphereObstacle(name="obst", contentDict=obstDict)
    n = 2
    q = ca.SX.sym("q", n)
    qdot = ca.SX.sym("qdot", n)
    var_q = Variables(state_variables={'q': q, 'qdot': qdot})
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    var_x = Variables(state_variables={'x': x, 'xdot': xdot})
    l_base = 1.0 * ca.dot(qdot, qdot)
    h_base = ca.SX(np.zeros(n))
    baseGeo = Geometry(h=h_base, var=var_q)
    baseLag = Lagrangian(l_base, var=var_q)
    planner = FabricPlanner(baseGeo, baseLag)
    phi = ca.norm_2(q - obst.position()) / obst.radius() - 1
    dm = DifferentialMap(phi, var=var_q)
    s = -0.5 * (ca.sign(xdot) - 1)
    lam = 5.00
    le = lam * 1/x * s * xdot**2
    lag_col = Lagrangian(le, var=var_x)
    h = -lam / (x ** 3) * xdot**2
    geo = Geometry(h=h, var=var_x)
    planner.addGeometry(dm, lag_col, geo)
    l_ex = 0.5 * ca.dot(qdot, qdot)
    exLag = Lagrangian(l_ex, var=var_q)
    exLag.concretize()
    planner.setExecutionEnergy(exLag)
    planner.concretize()
    # setup environment
    x0 = np.array([2.3, 0.5])
    xdot0 = np.array([-1.0, -0.0])
    # running the simulation
    env = gym.make('point-robot-acc-v0', dt=0.01, render=render)
    ob = env.reset(pos=x0, vel=xdot0)
    env.add_obstacle(obst)
    print("Starting episode")
    for i in range(n_steps):
        if i % 100 == 0:
            print('time step : ', i)
        # t0 = time.time()
        action = planner.computeAction(q=ob['x'], qdot=ob['xdot'])
        _, _, en_ex = exLag.evaluate(q=ob['x'], qdot=ob['xdot'])
        print(f"Execution Energy : {en_ex}")
        ob, reward, done, info = env.step(action)
    return {}


if __name__ == "__main__":
    n_steps = 1200
    pointMassAvoidance(n_steps)
