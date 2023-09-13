import sys
import os
import time

import numpy as np

import panda_py
from panda_py import controllers


from forwardkinematics.urdfFks.generic_urdf_fk import GenericURDFFk

from mpscenes.goals.goal_composition import GoalComposition
from mpscenes.obstacles.sphere_obstacle import SphereObstacle

from fabrics.planner.parameterized_planner import ParameterizedFabricPlanner

def set_planner():
    goal_dict = {
        "subgoal0": {
            "weight": 1.0,
            "is_primary_goal": True,
            "indices": [0, 1, 2],
            "parent_link": "panda_link0",
            "child_link": "panda_hand",
            "desired_position": [0.1, -0.6, 0.4],
            "epsilon": 0.05,
            "type": "staticSubGoal",
        },
        "subgoal1": {
            "weight": 5.0,
            "is_primary_goal": False,
            "indices": [0, 1, 2],
            "parent_link": "panda_link7",
            "child_link": "panda_hand",
            "desired_position": [0.1, 0.0, 0.0],
            "epsilon": 0.05,
            "type": "staticSubGoal",
        }
    }
    goal = GoalComposition(name="goal", content_dict=goal_dict)
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    with open(absolute_path + "/panda_for_fk.urdf", "r", encoding="utf-8") as file:
        urdf = file.read()
    forward_kinematics = GenericURDFFk(
        urdf,
        rootLink="panda_link0",
        end_link="panda_link9",
    )
    planner = ParameterizedFabricPlanner(
        7,
        forward_kinematics,
    )
    collision_links = ['panda_link9', 'panda_link7', 'panda_link3', 'panda_link4']
    panda_limits = [
            [-2.8973, 2.8973],
            [-1.7628, 1.7628],
            [-2.8973, 2.8973],
            [-3.0718, -0.0698],
            [-2.8973, 2.8973],
            [-0.0175, 3.7525],
            [-2.8973, 2.8973]
        ]
    # The planner hides all the logic behind the function set_components.
    planner.set_components(
        collision_links=collision_links,
        goal=goal,
        number_obstacles=0,
        number_plane_constraints=0,
        limits=panda_limits,
    )
    planner.concretize(mode='vel', time_step=0.01)
    return planner, forward_kinematics


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise RuntimeError(f'Usage: python {sys.argv[0]} <robot-hostname>')
    planner, fk = set_planner()

    stiffness = np.array([600., 600., 600., 600., 250., 150., 50.]) / 10
    damping = np.array([50, 50, 50, 20, 20, 20, 10]) * 0.0
    k_i = np.array([5, 5, 5, 2, 2, 2, 1]) * 1.0
    alpha_fabrics = 1.0
    alpha_internal = 0.1
    vel_error_cum_max = np.array([0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.1]) * 3
    vel_error_cum_min = -vel_error_cum_max

    ctrl = controllers.JointVelocity(stiffness=stiffness, damping=damping)
    ctrl.set_ki(k_i)
    ctrl.set_vel_error_cum_max(vel_error_cum_max)
    ctrl.set_vel_error_cum_min(vel_error_cum_min)
    ctrl.set_filter(alpha_internal)
    panda = panda_py.Panda(sys.argv[1])
    panda.get_robot().set_collision_behavior(
        np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]) * 5,
        np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]) * 5,
        np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]) * 5,
        np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]) * 5,
        np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0]) * 5,
        np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0]) * 5,
        np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0]) * 5,
        np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0]) * 5
    )
    q_0 = panda.get_state().q
    panda.move_to_start()
    panda.start_controller(ctrl)
    runtime = 100
    qdot_des = np.zeros(7)


    x_goal_0=np.array([0.4, 0.6, 0.4])
    x_goal_1=np.array([0.0, 0., -0.107])
    x_goal_1=np.array([0.107, 0., 0.0])
    weight_goal_0=1
    weight_goal_1=5

    qdot_filtered = np.zeros(7)
    qs = []
    qdots = []
    qdots_filtered = []


    with panda.create_context(frequency=100, max_runtime=runtime) as ctx:
        while ctx.ok():
            #qdot_des[0] += 0.01 * np.cos(ctrl.get_time())
            state = panda.get_state()
            q=state.q
            qdot=state.dq
            qs.append(q)
            qdots.append(qdot)
            qdot_filtered = (1-alpha_fabrics) * qdot_filtered + alpha_fabrics * np.array(qdot)
            qdots_filtered.append(qdot_filtered)
            arguments = dict(
                q=q,
                qdot=qdot_filtered,
                x_goal_0=x_goal_0,
                weight_goal_0=weight_goal_0,
                x_goal_1=x_goal_1,
                weight_goal_1=weight_goal_1,
            )
            qdot_des = planner.compute_action(**arguments)
            fk_ee = fk.numpy(q, 'panda_link0', 'panda_link8', positionOnly=True)
            print(np.linalg.norm(fk_ee- x_goal_0))

            ctrl.set_control(qdot_des)

    np.save('q', np.array(qs))
    np.save('q_d', np.array(qdots))
    np.save('q_d_filt', np.array(qdots_filtered))
