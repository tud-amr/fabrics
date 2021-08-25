import gym
import pointRobot
import casadi as ca
import numpy as np

from optFabrics.planner.fabricPlanner import DefaultFabricPlanner
from optFabrics.planner.default_geometries import CollisionGeometry
from optFabrics.planner.default_energies import CollisionLagrangian, ExecutionLagrangian
from optFabrics.planner.default_maps import CollisionMap

from optFabrics.diffGeometry.diffMap import DifferentialMap, RelativeDifferentialMap
from optFabrics.diffGeometry.referenceTrajectory import ReferenceTrajectory

from obstacle import Obstacle, DynamicObstacle
from robotPlot import RobotPlot


def pointMassDynamicAvoidance(n_steps=500):
    # Define the robot
    n = 2
    env = gym.make('point-robot-acc-v0', dt=0.001)
    ## setting up the problem
    t = ca.SX.sym('t', 1)
    x_obst = ca.vertcat(-3.0 + (0.5 * t)**2, -t * 0.1 + 0.1 * t**2)
    x_obst_fun = ca.Function("x_obst_fun", [t], [x_obst])
    refTraj = ReferenceTrajectory(2, ca.SX(np.identity(2)), traj=x_obst, t=t)
    refTraj.concretize()
    obsts = [
                DynamicObstacle(x_obst_fun, 1.0)
            ]
    planner = DefaultFabricPlanner(n, m_base=1)
    q, qdot = planner.var()
    fks = [q]
    # collision avoidance
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    lag_col = CollisionLagrangian(x, xdot)
    geo_col = CollisionGeometry(x, xdot,lam=10)
    for obst in obsts:
        q_rel = ca.SX.sym('q_rel', 2)
        qdot_rel = ca.SX.sym('qdot_rel', 2)
        for fk in fks:
            if isinstance(obst, DynamicObstacle):
                phi_n = ca.norm_2(q_rel) / obst.r()  - 1
                dm_n = DifferentialMap(phi_n, q=q_rel, qdot=qdot_rel)
                dm_rel = RelativeDifferentialMap(q=q, qdot=qdot, refTraj=refTraj)
                planner.addGeometry(dm_rel, lag_col.pull(dm_n), geo_col.pull(dm_n))
            elif isinstance(obst, Obstacle):
                dm_col = CollisionMap(q, qdot, fk, obst.x(), obst.r())
                planner.addGeometry(dm_col, lag_col, geo_col)
    exLag = ExecutionLagrangian(q, qdot)
    exLag.concretize()
    planner.setExecutionEnergy(exLag)
    planner.concretize()
    # setup environment
    qs = []
    x0 = np.array([3.0, 0.5])
    xdot0 = np.array([-1.0, -0.0])
    # running the simulation
    ob = env.reset(x0, xdot0)
    print("Starting episode")
    q = np.zeros((n_steps, n))
    t = 0.0
    for i in range(n_steps):
        if i % 100 == 0:
            print('time step : ', i)
        t += env._dt
        q_p_t, qdot_p_t, qddot_p_t = refTraj.evaluate(t)
        action = planner.computeAction(ob[0:2], ob[2:4], q_p_t, qdot_p_t, qddot_p_t)
        _, _, en_ex = exLag.evaluate(ob[0:2], ob[2:4])
        #print(en_ex)
        # env.render()
        ob, reward, done, info = env.step(action)
        q[i, :] = ob[0:n]
    qs.append(q)
    res = {}
    res['qs'] = qs
    res['obsts'] = obsts
    res['dt'] = env._dt
    return res

if __name__ == "__main__":
    n_steps = 5500
    res = pointMassDynamicAvoidance(n_steps)
    qs = res['qs']
    obsts = res['obsts']
    ## Plotting the results
    fk_fun = lambda q : q
    robotPlot = RobotPlot(qs, fk_fun, 2, types=[0], dt=res['dt'])
    robotPlot.initFig(2, 2)
    robotPlot.addObstacle([0], obsts)
    robotPlot.plot()
    robotPlot.makeAnimation(n_steps)
    robotPlot.show()
