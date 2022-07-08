import gym
import time
import casadi as ca
import logging
import planarenvs.multi_point_robots
import sys


import numpy as np

from fabrics.diffGeometry.diffMap import DifferentialMap, DynamicDifferentialMap
from fabrics.diffGeometry.geometry import Geometry
from fabrics.helpers.variables import Variables

logging.basicConfig(level=logging.DEBUG)

number_agents = 2
fully_ignorant_agents = []
action_aware_agents = []
computation_order = [0, 1]
initial_position_noise = 0.0
integrated_fabrics = len(sys.argv) == 2
logging.info(f"{integrated_fabrics}")


def initalize_environment(render=True):
    """
    Initializes the simulation environment.

    Adds obstacles and goal visualizaion to the environment based and
    steps the simulation once.
    """
    env = gym.make(
        "multi-point-robots-acc-v0",
        dt=0.01,
        render=render,
        number_agents=number_agents,
    )
    qdot0 = np.zeros(2*number_agents)
    if number_agents == 4:
        q0 = np.array([-1.5, 0, 1.5, 0, 0, 1.5, 0, -1.5])
        qdot0 = np.array([1.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 1.0])
    if number_agents == 3:
        q0 = np.array([-1.5, 0, 1.5, 0, 0, 1.5])
        qdot0 = np.array([1.0, 0.0, -1.0, 0.0, 0.0, -1.0])
    if number_agents == 2:
        q0 = np.array([-1.5, 0.1, 1.5, 0.121])
        qdot0 = np.array([0.0, 0.0, -1.0, 0.0]) 
    q0 += np.random.random(2*number_agents) * initial_position_noise
    initial_observation = env.reset(pos=q0, vel=qdot0)
    # Definition of the obstacle.
    return (env, initial_observation)

def svar(name, dim):
    return ca.SX.sym(name, dim)

def set_geometry():
    x_rel = svar('x_rel', 2)
    xdot_rel = svar('xdot_rel', 2)
    x_geo = svar('x_geo', 1)
    xdot_geo = svar('xdot_geo', 1)
    radius_robot_0 = 0.2
    radius_robot_1 = 0.2

    var_rel = Variables(state_variables={'x_rel': x_rel, 'xdot_rel': xdot_rel})
    phi_obst = ca.norm_2(x_rel)/(radius_robot_0 + radius_robot_1) - 1
    dm_static = DifferentialMap(phi_obst, var_rel)



    geos_root = []
    x_roots = []
    x_refs = []
    for j in range(number_agents):
        x_root = svar(f'x_root_{j}', 2)
        xdot_root = svar(f'xdot_root_{j}', 2)
        xddot_root = svar(f'xdot_root_{j}', 2)
        x_roots.append([x_root, xdot_root, xddot_root])
        var_root_raw = Variables(
                state_variables={f'x_root_{j}': x_root, f'xdot_root_{j}': xdot_root},
        )
        geo_root_j = Geometry(h=ca.SX(np.zeros(2)), var=var_root_raw)
        var_geo = Variables(state_variables={'x_geo': x_geo, 'xdot_geo': xdot_geo})
        h_geo = -1 / (x_geo**(2)) * (-0.5 * (ca.sign(xdot_geo) - 1)) * xdot_geo**2
        #h_geo = -1 / (x_geo**2) * xdot_geo**2
        logging.info(f"h_geo {j} : {h_geo}")
        geo = Geometry(h=h_geo, var=var_geo)
        x_ref_j = []
        for i in range(number_agents-1):
            ref_names = [f"x_ref_{j}_{i}", f"xdot_ref_{j}_{i}", f"xddot_ref_{j}_{i}"]
            x_ref = svar(ref_names[0], 2)
            xdot_ref = svar(ref_names[1], 2)
            xddot_ref = svar(ref_names[2], 2)
            x_ref_j.append([x_ref, xdot_ref, xddot_ref])
            var_root = Variables(
                    state_variables={f'x_root_{j}': x_root, f'xdot_root_{j}': xdot_root},
                    parameters={ref_names[0]: x_ref, ref_names[1]: xdot_ref, ref_names[2]: xddot_ref},
            )
            dynamic_map = DynamicDifferentialMap(var_root, ref_names=ref_names)


            geo_j_i = geo.pull(dm_static).dynamic_pull(dynamic_map)
            geo_root_j += geo_j_i
        x_refs.append(x_ref_j)
        if integrated_fabrics and j==1:
            h_0 = geos_root[0]._h
            x_root_0 = x_roots[0][0]
            xdot_root_0 = x_roots[0][1]
            x_root_1 = x_roots[1][0]
            xdot_root_1 = x_roots[1][1]
            xddot_root_1 = x_roots[1][2]
            x_ref_0_0 = x_refs[0][0][0]
            xdot_ref_0_0 = x_refs[0][0][1]
            xddot_ref_0_0 = x_refs[0][0][2]
            h_1 = geo_root_j._h
            assert len(ca.symvar(h_1)) == 10
            xddot_0 = -h_0
            h_1_subst_h0 = ca.substitute(h_1, xddot_ref, -xddot_0)
            assert len(ca.symvar(h_1_subst_h0)) == 18
            h_1_subst_h0_x0 = ca.substitute(h_1_subst_h0, x_root_0, x_ref)
            assert len(ca.symvar(h_1_subst_h0_x0)) == 16
            h_1_subst_h0_x0_xdot0 = ca.substitute(h_1_subst_h0_x0, xdot_root_0, xdot_ref)
            assert len(ca.symvar(h_1_subst_h0_x0_xdot0)) == 14
            h_1_subst_h0_x0_xdot0_xref0 = ca.substitute(h_1_subst_h0_x0_xdot0, x_ref_0_0, x_root_1)
            assert len(ca.symvar(h_1_subst_h0_x0_xdot0_xref0)) == 12
            h_1_subst_h0_x0_xdot0_xref0_xdotref0 = ca.substitute(h_1_subst_h0_x0_xdot0_xref0, xdot_ref_0_0, xdot_root_1)
            assert len(ca.symvar(h_1_subst_h0_x0_xdot0_xref0_xdotref0)) == 10
            h_1_subst_h0_x0_xdot0_xref0_xdotref0_xddotref0 = ca.substitute(h_1_subst_h0_x0_xdot0_xref0_xdotref0, xddot_ref_0_0, xddot_root_1)
            assert len(ca.symvar(h_1_subst_h0_x0_xdot0_xref0_xdotref0_xddotref0)) == 10
            f_1 = -h_1_subst_h0_x0_xdot0_xref0_xdotref0_xddotref0 - xddot_root_1
            f_1_fun = ca.Function('f_1_fun', [xddot_root_1, x_root_1, xdot_root_1, x_ref, xdot_ref], [f_1])
            f_1_solver = ca.rootfinder('f_1', 'newton', f_1_fun)
            logging.info("Using rootfinder")
            geos_root.append(f_1_solver)
        else:
            geo_root_j.concretize()
            geos_root.append(geo_root_j)

    return geos_root





def compute_action_geo(
    geometry, robot_index, ob, actions_other_agents
):
    other_positions = {}
    other_velocities = {}
    other_accelerations = {}
    arguments = {}
    for i in computation_order:
        if i == robot_index:
            arguments[f'x_root_{i}'] = ob['x'][2*i:2*i+2]
            arguments[f'xdot_root_{i}'] = ob['xdot'][2*i:2*i+2]
        else:
            if robot_index in fully_ignorant_agents:
                other_positions[i] = np.ones(2) * 10000
                other_velocities[i] = np.zeros(2)
            else:
                other_positions[i] = ob['x'][2*i:2*i+2]
                other_velocities[i] = ob['xdot'][2*i:2*i+2]
            if robot_index in action_aware_agents and i in list(actions_other_agents.keys()):
                other_accelerations[i] = actions_other_agents[i]
            else:
                other_accelerations[i] = np.zeros(2)
            #other_velocities.append(np.zeros(2))
    other_positions_list = list(other_positions.values())
    other_velocities_list = list(other_velocities.values())
    other_accelerations_list = list(other_accelerations.values())
    for i in range(number_agents-1):
        arguments[f"x_ref_{robot_index}_{i}"] = other_positions_list[i]
        arguments[f"xdot_ref_{robot_index}_{i}"] = other_velocities_list[i]
        arguments[f"xddot_ref_{robot_index}_{i}"] = other_accelerations_list[i]

    t0 = time.perf_counter()
    if integrated_fabrics and robot_index == 1:
        function_args = [
                np.random.random(2),
                arguments[f'x_root_{robot_index}'],
                arguments[f'xdot_root_{robot_index}'],
                arguments[f'x_ref_{robot_index}_{0}'],
                arguments[f'xdot_ref_{robot_index}_{0}'],
        ]
        action = geometry(
                *function_args
        )
        action = np.array(action)[:,0]
    else:
        _, action = geometry.evaluate(
            **arguments
        )
    t1 = time.perf_counter()
    #logging.info(f"computation time : {(t1-t0)*1e3}")
    logging.debug(f"action for agent {robot_index} : {action}")
    #action = np.clip(action, np.ones(2) * -action_max, action_max * np.ones(2))
    return action

def compute_all_actions_geo(geometry, ob):
    actions = np.zeros(2*number_agents)
    action_list = {}
    for i in computation_order:
        action_list[i] = compute_action_geo(
                geometry[i],
                i,
                ob,
                action_list,
        )
    for i in computation_order:
        actions[2*i:2*(i+1)] = action_list[i]
    return actions



def run_point_robot_example(n_steps=5000, render=True):
    (env, initial_observation) = initalize_environment(render=render)
    ob = initial_observation
    geometries = set_geometry()

    # Start the simulation
    logging.info("Starting simulation")
    for i in range(n_steps):
        logging.debug(f"Current time {i}")
        action = compute_all_actions_geo(geometries, ob)
        ob, *_ = env.step(action)
    return {}


if __name__ == "__main__":
    res = run_point_robot_example(n_steps=10000)
