import gym
import nLinkReacher
import time
import casadi as ca
import numpy as np

from optFabrics.planner.fabricPlanner import DefaultFabricPlanner
from optFabrics.planner.default_geometries import CollisionGeometry, GoalGeometry
from optFabrics.planner.default_energies import CollisionLagrangian, ExecutionLagrangian, GoalLagrangian
from optFabrics.planner.default_maps import CollisionMap, VariableGoalMap, GoalMap
from optFabrics.planner.default_leaves import defaultAttractor

from optFabrics.diffGeometry.diffMap  import DifferentialMap, VariableDifferentialMap, RelativeDifferentialMap
from optFabrics.diffGeometry.energy import Lagrangian
from optFabrics.diffGeometry.geometry import Geometry
from optFabrics.diffGeometry.energized_geometry import WeightedGeometry


from obstacle import Obstacle, DynamicObstacle
from robotPlot import RobotPlot
from solverPlot import SolverPlot
from casadiFk import casadiFk
from numpyFk import numpyFk


def nlinkDynamicGoal(n=3, n_steps=5000):
    env = gym.make('nLink-reacher-acc-v0', n=n, dt=0.01)
    t = ca.SX.sym('t', 1)
    w_obst = 0.2
    x_obst = ca.vertcat(-1.2, -3.0 * ca.sin(w_obst * t))
    v_obst = ca.jacobian(x_obst, t)
    a_obst = ca.jacobian(v_obst, t)
    x_obst_fun = ca.Function("x_obst_fun", [t], [x_obst])
    v_obst_fun = ca.Function("v_obst_fun", [t], [v_obst])
    a_obst_fun = ca.Function("a_obst_fun", [t], [a_obst])
    r = 1.0
    obsts = [
                #DynamicObstacle(x_obst_fun, r),
                #Obstacle(np.array([-1.0, 2.5]), 0.5)
            ]
    planner = DefaultFabricPlanner(n, m_base=0.1)
    q, qdot = planner.var()
    # collision avoidance
    x = ca.SX.sym("x", 1)
    xdot = ca.SX.sym("xdot", 1)
    lag_col = CollisionLagrangian(x, xdot)
    geo_col = CollisionGeometry(x, xdot)
    fks = []
    for i in range(1, n + 1):
        fks.append(casadiFk(q, i)[0:2])
    for obst in obsts:
        q_p = ca.SX.sym('q_p', 2)
        qdot_p = ca.SX.sym('qdot_p', 2)
        qddot_p = ca.SX.sym('qddot_p', 2)
        for fk in fks:
            if isinstance(obst, DynamicObstacle):
                dm_col = VariableCollisionMap(q, qdot, fk, obst.r(), q_p, qdot_p, qddot_p)
            elif isinstance(obst, Obstacle):
                dm_col = CollisionMap(q, qdot, fk, obst.x(), obst.r())
            planner.addGeometry(dm_col, lag_col, geo_col)
    """
    # constant forcing in joint space
    q_d = np.array([0.0, 0.0, 0.0])
    psi = 25 * ca.norm_2(q - q_d)**2
    h_psi = ca.gradient(psi, q)
    geo_psi = Geometry(h=h_psi, x=q, xdot=qdot)
    l_psi = 0.5 * ca.dot(qdot, qdot)
    lag_psi = Lagrangian(l_psi, x=q, xdot=qdot)
    dm_psi = DifferentialMap(q, q=q, qdot=qdot)
    planner.addForcingGeometry(dm_psi, lag_psi, geo_psi)
    """
    """
    # constant forcing in joint space
    x = ca.SX.sym("x", 2)
    xdot = ca.SX.sym("xdot", 2)
    x_d = np.array([-1.0, 0.0])
    psi = 1 * ca.norm_2(x)**2
    h_psi = ca.gradient(psi, x)
    geo_psi = Geometry(h=h_psi, x=x, xdot=xdot)
    l_psi = 0.5 * ca.dot(xdot, xdot)
    lag_psi = Lagrangian(l_psi, x=x, xdot=xdot)
    dm_psi = DifferentialMap(fks[-1]-x_d, q=q, qdot=qdot)
    """
    """
    # variant forcing in joint space
    t = ca.SX.sym("t", 1)
    w = 0.3
    q_g = ca.vertcat(2.0 * ca.sin(w * t), 0.0, 0.0)
    qdot_g = ca.jacobian(q_g, t)
    qddot_g = ca.jacobian(qdot_g, t)
    q_g_fun = ca.Function("q_g", [t], [q_g])
    qdot_g_fun = ca.Function("qdot_g", [t], [qdot_g])
    qddot_g_fun = ca.Function("qddot_g", [t], [qddot_g])
    q_d = ca.SX.sym("q_d", 3)
    qdot_d = ca.SX.sym("qdot_d", 3)
    qddot_d = ca.SX.sym("qddot_d", 3)
    q_rel = ca.SX.sym("q_rel", 3)
    qdot_rel = ca.SX.sym("qdot_rel", 3)
    qddot_rel = ca.SX.sym("qddot_rel", 3)
    psi = 25 * ca.norm_2(q_rel)**2
    h_psi = ca.gradient(psi, q_rel)
    geo_psi = Geometry(h=h_psi, x=q_rel, xdot=qdot_rel)
    dm_rel = RelativeDifferentialMap(q=q, qdot=qdot, q_p=q_d, qdot_p=qdot_d, qddot_p=qddot_d)
    # first energize and then pull
    l_rel = 0.5 * ca.dot(qdot_rel, qdot_rel)
    lag_rel = Lagrangian(l_rel, x=q_rel, xdot=qdot_rel)
    eg_psi = WeightedGeometry(g=geo_psi, le=lag_rel)
    eg_psi_fin = eg_psi.pull(dm_rel)
    # first pull and then energize
    l_psi = 0.5 * ca.dot(qdot, qdot)
    lag_psi = Lagrangian(l_psi, x=q, xdot=qdot)
    geo_psi_fin = geo_psi.pull(dm_rel)
    #eg_psi_fin = WeightedGeometry(g=geo_psi_fin, le=lag_psi)

    dm_psi = DifferentialMap(q, q=q, qdot=qdot)
    eg_psi_fin.concretize()
    planner.addForcingWeightedGeometry(dm_psi, eg_psi_fin, goalVelocity=qdot_d)
    """
    # variant forcing in joint space
    t = ca.SX.sym("t", 1)
    w = 0.9
    x = ca.SX.sym('x', 2)
    xdot = ca.SX.sym('xdot', 2)
    xddot = ca.SX.sym('xddot', 2)
    x_g = ca.vertcat(1.8 + 0.5 * ca.cos(w * t),  1.0 * ca.sin(w * t))
    x_goal = ca.Function("x_goal", [t], [x_g])
    xdot_g = ca.jacobian(x_g, t)
    xddot_g = ca.jacobian(xdot_g, t)
    x_g_fun = ca.Function("x_g", [t], [x_g])
    xdot_g_fun = ca.Function("xdot_g", [t], [xdot_g])
    xddot_g_fun = ca.Function("xddot_g", [t], [xddot_g])
    x_d = ca.SX.sym("x_d", 2)
    xdot_d = ca.SX.sym("xdot_d", 2)
    xddot_d = ca.SX.sym("xddot_d", 2)
    x_rel = ca.SX.sym("x_rel", 2)
    xdot_rel = ca.SX.sym("xdot_rel", 2)
    xddot_rel = ca.SX.sym("xddot_rel", 2)
    psi = 10 * ca.norm_2(x_rel)**2
    h_psi = ca.gradient(psi, x_rel)
    geo_psi = Geometry(h=h_psi, x=x_rel, xdot=xdot_rel)
    dm_rel = RelativeDifferentialMap(q=x, qdot=xdot, q_p=x_d, qdot_p=xdot_d, qddot_p=xddot_d)
    # first energize and then pull
    l_rel = 0.5 * ca.dot(xdot_rel, xdot_rel)
    lag_rel = Lagrangian(l_rel, x=x_rel, xdot=xdot_rel)
    eg_psi = WeightedGeometry(g=geo_psi, le=lag_rel)
    eg_psi_fin = eg_psi.pull(dm_rel)
    # first pull and then energize
    l_psi = 0.5 * ca.dot(xdot, xdot)
    lag_psi = Lagrangian(l_psi, x=x, xdot=xdot)
    geo_psi_fin = geo_psi.pull(dm_rel)
    #eg_psi_fin = WeightedGeometry(g=geo_psi_fin, le=lag_psi)

    dm_psi = DifferentialMap(fks[-1], q=q, qdot=qdot)
    eg_psi_fin.concretize()
    eg_psi_fin_pull = eg_psi_fin.pull(dm_psi)
    planner.addForcingWeightedGeometry(dm_psi, eg_psi_fin, goalVelocity=xdot_d)

    # add forcing term
    #planner.addForcingGeometry(dm_psi, lag_psi, geo_psi)
    eg_f = planner._eg_f
    eg_f.concretize()
    """
    # forcing term variable
    t = ca.SX.sym("t", 1)
    w = 2.0
    x_d = ca.vertcat(1.5 + 0.7 * ca.sin(w * t), 2 * ca.cos(w * t))
    #x_d = 2 * ca.vertcat(ca.cos(w * t), 1.0 * ca.sin(w * t))
    x_goal = ca.Function('x_goal', [t], [x_d])
    v_d = ca.jacobian(x_d, t)
    a_d = ca.jacobian(v_d, t)
    x_d_fun = ca.Function("x_d_fun", [t], [x_d])
    v_d_fun = ca.Function("v_d_fun", [t], [v_d])
    a_d_fun = ca.Function("a_d_fun", [t], [a_d])
    # variables
    x = ca.SX.sym("x", 2)
    xdot = ca.SX.sym("xdot", 2)
    x_g = ca.SX.sym("x_g", 2)
    xdot_g = ca.SX.sym("xdot_g", 2)
    xddot_g = ca.SX.sym("xdot_g", 2)
    x_rel = ca.SX.sym("x_rel", 2)
    xdot_rel = ca.SX.sym("xdot_rel", 2)
    # relative systems
    l_psi = 0.5 * ca.dot(xdot, xdot)
    lag_psi = Lagrangian(l_psi, x=x, xdot=xdot)
    k_psi = 20
    psi_rel = k_psi * ca.norm_2(x_rel)**2
    h_rel = ca.gradient(psi_rel, x_rel)
    geo_rel = Geometry(h=h_rel, x=x_rel, xdot=xdot_rel)
    dm_rel = RelativeDifferentialMap(q=x, qdot=xdot, q_p=x_g, qdot_p=xdot_g, qddot_p=xddot_g)
    geo_psi = geo_rel.pull(dm_rel)
    phi_psi = fks[-1]
    dm_psi = DifferentialMap(phi_psi, q=q, qdot=qdot)
    #planner.addForcingGeometry(dm_psi, lag_psi, geo_psi, goalVelocity=xdot_g)
    """
    # execution energy
    exLag = ExecutionLagrangian(q, qdot)
    planner.setExecutionEnergy(exLag)
    # Speed control
    ex_factor = 1.0
    planner.setDefaultSpeedControl(x, dm_psi, exLag, ex_factor)
    #planner.setConstantSpeedControl(beta=1.0)
    planner.concretize()
    # setup environment
    qs = []
    solverTimes = []
    # running the simulation
    for e in range(2):
        ob = env.reset()
        print("Starting episode")
        q = np.zeros((n_steps, n))
        t = 0.0
        solverTime = np.zeros(n_steps)
        for i in range(n_steps):
            #print("------------------------------------------")
            if i % 1000 == 0:
                print('time step : ', i)
            t += env._dt
            q_p_t = np.array(x_obst_fun(t))[:, 0]
            qdot_p_t = np.array(v_obst_fun(t))[:, 0]
            qddot_p_t = np.array(a_obst_fun(t))[:, 0]
            q_g_t = np.array(x_g_fun(t))[:, 0]
            qdot_g_t = np.array(xdot_g_fun(t))[:, 0]
            qddot_g_t = np.array(xddot_g_fun(t))[:, 0]
            """
            print("q_g_t : ", q_g_t)
            print("qdot_g_t : ", qdot_g_t)
            print("qddot_g_t : ", qddot_g_t)
            """
            if e == 1:
                if i == 0:
                    print('run %i with no relative part' % e)
                qdot_g_t = np.zeros(2)
                qddot_g_t = np.zeros(2)
            t0 = time.time()
            #action = planner.computeAction(ob[0:n], ob[n:2*n], q_p_t, qdot_p_t, q_g_t, qdot_g_t)
            q_t = ob[0:n]
            qdot_t = ob[n:2*n]
            """
            x_t, J_t, Jdot_t = dm_psi.forward(q_t, qdot_t)
            xdot_t = np.dot(J_t, qdot_t)
            print("state : ", q_t)
            #action = planner.computeAction(q_t, qdot_t, q_g_t, qdot_g_t, qddot_g_t)
            h_psi, xddot_psi = geo_psi.evaluate(x_t, xdot_t)
            print("x_t : ", x_t)
            print("xdot_t : ", xdot_t)
            print("h_psi : ", h_psi)
            M, f, xddot, alpha = eg_f.evaluate(ob[0:n], ob[n:2*n], q_g_t, qdot_g_t, qddot_g_t)
            print("M : ", M)
            print("f : ", f)
            #f_test = 2 * 25 * (q_t - q_g_t) - qddot_g_t
            #print("f_test: ", f_test)
            """
            #action = planner.computeAction(ob[0:n], ob[n:2*n], q_p_t, qdot_p_t, qddot_p_t, q_g_t, qdot_g_t, qddot_g_t)
            action = planner.computeAction(ob[0:n], ob[n:2*n], q_g_t, qdot_g_t, qddot_g_t)
            #action = planner.computeAction(ob[0:n], ob[n:2*n])
            #print("action : ", action)
            solverTime[i] = time.time() - t0
            # env.render()
            ob, reward, done, info = env.step(action)
            q[i, :] = ob[0:n]
        qs.append(q)
        solverTimes.append(solverTime)
    ## Plotting the results
    res = {}
    res['qs'] = qs
    res['solverTimes'] = solverTimes
    res['obsts'] = obsts
    res['dt'] = env._dt
    res['x_d'] = x_goal
    return res


if __name__ == "__main__":
    n_steps = 10000
    n = 3
    res = nlinkDynamicGoal(n=n, n_steps=n_steps)
    if n_steps > 500:
        fk_fun = lambda q, n: numpyFk(q, n)[0:3]
        sol_indices = [0, 1]
        robotPlot = RobotPlot([res['qs'][i] for i in sol_indices], fk_fun, 2, types=[1, 1], dt=res['dt'])
        robotPlot.initFig(2, 2)
        robotPlot.addObstacle([0, 1], res['obsts'])
        robotPlot.plot()
        robotPlot.makeAnimation(n_steps)
        #robotPlot.addSolutions(0, res['qs'])
        robotPlot.addGoal([0, 1], res['x_d'])
        solverAxs = robotPlot.getAxs([2, 3])
        solverPlot = SolverPlot([res['solverTimes'][i] for i in sol_indices], 1, 1, axs=solverAxs)
        solverPlot.plot()
        robotPlot.show()
