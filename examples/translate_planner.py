from typing import Callable
import time
from copy import deepcopy

import numpy as np

from fabrics.helpers.translation import c2np

"""
Use the example point_robot_urdf.py to generate a simple planner
```python
planner.export_as_c(simple_planner.c')
```

Then you can use this example to test the numpy function for parallelization.
"""

python_code = c2np('simple_planner.c', 'simple_planner.py' )
from simple_planner import casadi_f0_numpy

def forward_simulate(function: Callable, q0: np.ndarray, qdot0: np.ndarray, n_steps: int) -> np.ndarray:
    dt = 0.01
    q = np.transpose(q0)[0]
    qdot = np.transpose(qdot0)[0]
    trajectory = []
    for i in range(n_steps):
        action = function(np.reshape(q, (3, 1)), np.reshape(qdot, (3, 1)))[0]
        qdot = np.transpose(action)
        q += qdot * dt
        trajectory.append(deepcopy(q))
    return np.array(trajectory)

N = 1

q0 = np.array([[-1.9998125,0.5, 0]])
qdot0 = np.array([[0.0187499776, 0.0, 0]])
x_goal = np.array([[3.5, 0.5]])

weight_goal_0 = np.repeat(np.array([0.5]), N)
radius_obst_0 = np.repeat(np.array([1.0]), N)
radius_body = np.repeat(np.array([0.2]), N)

q = np.repeat(np.transpose(q0), N, axis=1)
qdot = np.repeat(np.transpose(qdot0), N, axis=1)
x_obst = np.array([[2.0, 0.0, 0.0]])
x_obst_0 = np.repeat(np.transpose(x_obst), N, axis=1)

x_goal_0 = np.repeat(np.transpose(x_goal), N, axis=1)
"""
x_goal_0 = np.transpose(np.array([
    [3.5, 0.5],
    [3.5, 1.5],
    [-3.5, 0.5],
    [1.5, 2.0],
    [3.5, -0.5],
]))
"""

t0 = time.perf_counter()
res2 = casadi_f0_numpy(q, qdot, radius_body, radius_obst_0, weight_goal_0, x_goal_0, x_obst_0)
t1 = time.perf_counter()
print(res2)
print(f'Computed {N} actions in {(t1-t0)} which means {1000* (t1-t0)/N} ms for action')


def for_fun(q, qdot):
    N = 1
    x_goal = np.array([[3.5, 0.5]])

    weight_goal_0 = np.repeat(np.array([0.5]), N)
    radius_obst_0 = np.repeat(np.array([1.0]), N)
    radius_body = np.repeat(np.array([0.2]), N)

    x_obst = np.array([[2.0, 0.0, 0.0]])
    x_obst_0 = np.repeat(np.transpose(x_obst), N, axis=1)

    x_goal_0 = np.repeat(np.transpose(x_goal), N, axis=1)

    return casadi_f0_numpy(q, qdot, radius_body, radius_obst_0, weight_goal_0, x_goal_0, x_obst_0)

t0 = time.perf_counter()
traj = forward_simulate(for_fun, q, qdot, 1000)
t1 = time.perf_counter()
print(t1-t0)





