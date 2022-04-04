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

from fabrics.helpers.variables import Variables

from MotionPlanningEnv.sphereObstacle import SphereObstacle
from MotionPlanningEnv.dynamicSphereObstacle import DynamicSphereObstacle

"""
Maintains energy despite moving obstacle.
"""


def pointMassDynamicAvoidance(n_steps=500, render=True):
    # Define the robot
    n = 2
    env = gym.make('point-robot-acc-v0', dt=0.001, render=render)
    # setting up the problem
    obst_traj = ['-3.0 + (0.5 * t)**2', '-t * 0.1 + 0.1 * t**2']
    x_obst = ca.SX.sym("x_obst", 2)
    xdot_obst = ca.SX.sym("xdot_obst", 2)
    xddot_obst = ca.SX.sym("xddot_obst", 2)
    var_obst = Variables(parameters={'x_obst': x_obst, 'xdot_obst': xdot_obst, 'xddot_obst': xddot_obst})
    refTraj = AnalyticSymbolicTrajectory(ca.SX(np.identity(2)), 2, var=var_obst, traj=obst_traj)
    refTraj.concretize()
    obstDict = {'dim': 2, 'type': 'sphere', 'geometry': {'position': [1.0, -2.0], 'radius': 0.2}} 
    obst2Dict = {'dim': 2, 'type': 'sphere', 'geometry': {'trajectory': obst_traj, 'radius': 1.2}} 
    obsts = [
                SphereObstacle(name="obst1", contentDict=obstDict), 
                DynamicSphereObstacle(name="obst2", contentDict=obst2Dict)
            ]
    planner = DefaultFabricPlanner(n, m_base=1)
    var_q = planner.var()
    fks = [var_q.position_variable()]
    # collision avoidance
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    var_x = Variables(state_variables={'x': x, 'xdot': xdot})
    lag_col = CollisionLagrangian(var_x)
    geo_col = CollisionGeometry(var_x,lam=10)
    for i, obst in enumerate(obsts):
        x_col = ca.SX.sym("x_col", 2)
        xdot_col = ca.SX.sym("xdot_col", 2)
        var_col = Variables(state_variables={'x_col': x_col, 'xdot_col': xdot_col})
        x_rel = ca.SX.sym("x_rel", 2)
        xdot_rel = ca.SX.sym("xdot_rel", 2)
        var_rel = Variables(state_variables={'x_rel': x_rel, 'xdot_rel': xdot_rel})
        for fk in fks:
            if isinstance(obst, DynamicSphereObstacle):
                phi_n = ca.norm_2(x_rel) / obst.radius() - 1
                dm_n = DifferentialMap(phi_n, var=var_rel)
                dm_rel = RelativeDifferentialMap(var=var_col, refTraj=refTraj)
                dm_col = DifferentialMap(fk, var=var_q)
                planner.addGeometry(dm_col, lag_col.pull(dm_n).pull(dm_rel), geo_col.pull(dm_n).pull(dm_rel))
            elif isinstance(obst, SphereObstacle):
                dm_col = CollisionMap(var_q, fk, obst.position(), obst.radius())
                planner.addGeometry(dm_col, lag_col, geo_col)
    exLag = ExecutionLagrangian(var_q)
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
            q=ob['x'], qdot=ob['xdot'],
            x_obst=q_p_t, xdot_obst=qdot_p_t, xddot_obst=qddot_p_t
        )
        _, _, en_ex = exLag.evaluate(q=ob['x'], qdot=ob['xdot'])
        print(en_ex)
        ob, reward, done, info = env.step(action)
    return {}


if __name__ == "__main__":
    n_steps = 10000
    pointMassDynamicAvoidance(n_steps)
