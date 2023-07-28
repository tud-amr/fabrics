import casadi as ca
import numpy as np
from fabrics.diffGeometry.diffMap import DifferentialMap
from fabrics.diffGeometry.energy import Lagrangian
from fabrics.diffGeometry.geometry import Geometry
from fabrics.helpers.variables import Variables
from fabrics.diffGeometry.energized_geometry import WeightedGeometry
import gym
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import planarenvs.point_robot
import sys
from mpscenes.obstacles.sphere_obstacle import SphereObstacle

plt.rcParams['figure.dpi'] = 200

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


def run_environment(
        weighted_geometry,
        render=False,
        dt=0.001,
        T=20,
        obstacles=[],
        init_vel=[1.0, 0.1],
        init_pos=[0.5, 0.1],
):
    # Using openai gym interface for the simulation
    env = gym.make("point-robot-acc-v0", render=render, dt=dt)
    n_steps = int(T / dt)

    positions = []
    ob = env.reset(pos=np.array(init_pos), vel=np.array(init_vel))
    for obstacle in obstacles:
        env.add_obstacle(obstacle)
    for i in range(n_steps):
        x = ob['joint_state']['position']
        xdot = ob['joint_state']['velocity']
        positions.append(x)

        # Simple logging to monitor the proces
        if i % (1 / dt) == 0:
            print(f"Current time : {i * dt}/{T}")

            # Use a weighted geometry to compute the action
        _, _, xddot, alpha = weighted_geometry.evaluate(q=x, qdot=xdot)
        action = xddot - alpha * xdot

        # Break if something goes horribly wrong
        if np.linalg.norm(action) > 1e5:
            print(f"Breaking due to action magnitude at step {dt * i}, {x}")
            break
        if np.any(np.isnan(action)):
            print(f"Breaking due to isnan at step {dt * i}, {x}")
            break
        if np.linalg.norm(xdot) > 1.2 * np.linalg.norm(np.array(init_vel)):
            print(f"Breaking due to velocity magnitude at step {dt * i}, {x}")
            break

        ob, _, _, _ = env.step(action)
    env.close()
    return np.array(positions)


def plot_trajectory(positions, qmax=None, obstacles=[]):
    fig, axs = plt.subplots(1, 1)
    axs.plot(positions[:, 0], positions[:, 1])
    axs.plot(positions[0, 0], positions[0, 1], "k*", markersize=10, label='start point')
    axs.axis('on')

    for obstacle in obstacles:
        axs.add_patch(plt.Circle(obstacle.position(), radius=obstacle.radius(), color='r', fill=None, label='obstacle'))
    axs.axis('equal')

    if qmax:
        ell = Ellipse(xy=[0, 0], width=qmax[0] * 2.02, height=qmax[1] * 2.02, angle=0,
                      edgecolor='r', lw=1, facecolor='none')
        axs.add_artist(ell)
        axs.set_xlim([-1.3 * qmax[0], 1.3 * qmax[0]])
        axs.set_ylim([-1.3 * qmax[1], 1.3 * qmax[1]])

    plt.legend()
    return plt


def main():
    # create a base geometry
    q = ca.SX.sym('q', 2)
    qdot = ca.SX.sym('qdot', 2)
    config_variables = Variables(state_variables={'q': q, 'qdot': qdot})
    geo_b = Geometry(var=config_variables, h=ca.SX(np.zeros(2)))
    lag_b = Lagrangian(0.5 * ca.norm_2(qdot) ** 2, var=config_variables)
    fabric_b = WeightedGeometry(le=lag_b, g=geo_b)
    fabric_b.concretize()
    # add a circle obstacle
    obstacles = []
    circle_dict = {
        'type': 'SphereObstacle',
        'geometry': {
            'position': [0.7, -0.1],
            'radius': 0.3,
        }
    }
    obstacles.append(SphereObstacle(name="CircleObstacle", content_dict=circle_dict))
    # collision avoidance
    x = ca.SX.sym("x_obst", 1)
    xdot = ca.SX.sym("xdot_obst", 1)
    variables = Variables(state_variables={'x': x, 'xdot': xdot})

    h_obst = -2 / (x ** 2) * xdot ** 2 * (-0.5 * (ca.sign(xdot) - 1))
    le = 2 / (x ** 2) * xdot ** 2

    geometry_obstacle = Geometry(var=variables, h=h_obst)
    le_obst = Lagrangian(le, var=variables)
    weighted_geometry = WeightedGeometry(le=le_obst, g=geometry_obstacle)
    phi = ca.norm_2(q - obstacles[0].position()) / obstacles[0].radius() - 1
    diff_map = DifferentialMap(phi, config_variables)
    fabric_o = weighted_geometry.pull(diff_map)
    fabric_o_b = fabric_o + fabric_b

    l_exec = Lagrangian(0.5 * ca.norm_2(qdot) ** 2, var=config_variables)
    final_geo = WeightedGeometry(g=Geometry(s=fabric_o_b), le=l_exec)
    final_geo.concretize()
    positions_o_b = run_environment(final_geo, T=2, init_pos=[0.0, 0.0], init_vel=[1.0, 0.0])

    plot_trajectory(positions_o_b, obstacles=obstacles).show()


if __name__ == "__main__":
    main()
