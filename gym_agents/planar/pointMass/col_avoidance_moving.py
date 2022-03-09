import gym
import planarenvs.pointRobot
import casadi as ca
import numpy as np

from fabrics.planner.fabricPlanner import DefaultFabricPlanner
from fabrics.planner.default_geometries import CollisionGeometry
from fabrics.planner.default_energies import CollisionLagrangian, ExecutionLagrangian
from fabrics.planner.default_maps import CollisionMap

from fabrics.diffGeometry.diffMap import DifferentialMap, RelativeDifferentialMap
from fabrics.diffGeometry.analyticSymbolicTrajectory import AnalyticSymbolicTrajectory

from MotionPlanningEnv.sphereObstacle import SphereObstacle
from MotionPlanningEnv.dynamicSphereObstacle import DynamicSphereObstacle


def pointMassDynamicAvoidance(n_steps=500, render=True):
    # Define the robot
    n = 2
    env = gym.make('point-robot-acc-v0', dt=0.001, render=render)
    # setting up the problem
    x_obst = ['-3.0 + (0.5 * t)**2', '-t * 0.1 + 0.1 * t**2']
    refTraj = AnalyticSymbolicTrajectory(ca.SX(np.identity(2)), 2, traj=x_obst)
    refTraj.concretize()
    obstDict = {'dim': 2, 'type': 'sphere', 'geometry': {'position': [1.0, -2.0], 'radius': 0.2}} 
    obst2Dict = {'dim': 2, 'type': 'sphere', 'geometry': {'trajectory': x_obst, 'radius': 1.2}} 
    obsts = [
                SphereObstacle(name="obst1", contentDict=obstDict), 
                DynamicSphereObstacle(name="obst2", contentDict=obst2Dict)
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
            if isinstance(obst, DynamicSphereObstacle):
                phi_n = ca.norm_2(q_rel) / obst.radius()  - 1
                dm_n = DifferentialMap(phi_n, q=q_rel, qdot=qdot_rel)
                dm_rel = RelativeDifferentialMap(q=q, qdot=qdot, refTraj=refTraj)
                planner.addGeometry(dm_rel, lag_col.pull(dm_n), geo_col.pull(dm_n))
            elif isinstance(obst, SphereObstacle):
                dm_col = CollisionMap(q, qdot, fk, obst.position(), obst.radius())
                planner.addGeometry(dm_col, lag_col, geo_col)
    exLag = ExecutionLagrangian(q, qdot)
    exLag.concretize()
    planner.setExecutionEnergy(exLag)
    planner.concretize()
    # setup environment
    x0 = np.array([3.0, 0.5])
    xdot0 = np.array([-1.0, -0.0])
    # running the simulation
    ob = env.reset(pos=x0, vel=xdot0)
    for obst in obsts:
        env.addObstacle(obst)
    print("Starting episode")
    for i in range(n_steps):
        if i % 1000 == 0:
            print('time step : ', i)
        q_p_t, qdot_p_t, qddot_p_t = refTraj.evaluate(env.t())
        action = planner.computeAction(
            ob['x'], ob['xdot'],
            q_p_t, qdot_p_t, qddot_p_t
        )
        _, _, en_ex = exLag.evaluate(ob['x'], ob['xdot'])
        print(en_ex)
        ob, reward, done, info = env.step(action)
    return {}


if __name__ == "__main__":
    n_steps = 10000
    pointMassDynamicAvoidance(n_steps)
