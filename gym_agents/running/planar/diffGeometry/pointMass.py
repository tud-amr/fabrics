import gym
import pointRobot
import time
import casadi as ca
import numpy as np

from optFabrics.planner.fabricPlanner import DefaultFabricPlanner
from optFabrics.planner.default_geometries import CollisionGeometry, GoalGeometry
from optFabrics.planner.default_energies import CollisionLagrangian, ExecutionLagrangian, GoalLagrangian
from optFabrics.planner.default_maps import CollisionMap
from optFabrics.planner.default_leaves import defaultAttractor

from MotionPlanningEnv.sphereObstacle import SphereObstacle


def pointMass(n_steps=5000):
    ## setting up the problem
    staticObstDict = {'dim': 2, 'type': 'sphere', 'geometry': {'position': [0.0, 0.0], 'radius': 1.0}} 
    obsts = [
        SphereObstacle(name="staticObst", contentDict=staticObstDict),
    ]
    n = 2
    planner = DefaultFabricPlanner(n)
    q, qdot = planner.var()
    # collision avoidance
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    lag_col = CollisionLagrangian(x, xdot)
    geo_col = CollisionGeometry(x, xdot, exp=2.0)
    fks = [q]
    for fk in fks:
        for obst in obsts:
            dm_col = CollisionMap(q, qdot, fk, obst.position(), obst.radius())
            planner.addGeometry(dm_col, lag_col, geo_col)
    # forcing term
    q_d = np.array([-2.0, -0.1])
    dm_psi, lag_psi, _, x_psi, xdot_psi  = defaultAttractor(q, qdot, q_d, fk)
    geo_psi = GoalGeometry(x_psi, xdot_psi, k_psi=5)
    planner.addForcingGeometry(dm_psi, lag_psi, geo_psi)
    # execution energy
    exLag = ExecutionLagrangian(q, qdot)
    exLag.concretize()
    planner.setExecutionEnergy(exLag)
    # Speed control
    ex_factor = 1.0
    planner.setDefaultSpeedControl(x_psi, dm_psi, exLag, ex_factor)
    planner.concretize()
    # setup environment
    qs = []
    solverTimes = []
    x0s = [np.array([4.3, -1.0 + i * 0.2]) for i in range(11)]
    xdot0s = [np.array([-1.0, -0.0])]
    # running the simulation
    for xdot0 in xdot0s:
        for x0 in x0s:
            env = gym.make('point-robot-acc-v0', dt=0.01, render=True)
            ob = env.reset(x0, xdot0)
            for obst in obsts:
                env.addObstacle(obst)
            print("Starting episode")
            q = np.zeros((n_steps, n))
            t = 0.0
            solverTime = np.zeros(n_steps)
            for i in range(n_steps):
                """
                if i % 100 == 0:
                    print('time step : ', i)
                """
                t += env._dt
                t0 = time.time()
                action = planner.computeAction(ob['x'], ob['xdot'])
                solverTime[i] = time.time() - t0
                ob, reward, done, info = env.step(action)
                q[i, :] = ob['x']
            qs.append(q)
            solverTimes.append(solverTime)
    ## Plotting the results
    res = {}
    res['qs'] = qs
    res['solverTimes'] = solverTimes
    res['obsts'] = obsts
    res['dt'] = env._dt
    return res

if __name__ == "__main__":
    n_steps = 5000
    res = pointMass(n_steps=n_steps)
