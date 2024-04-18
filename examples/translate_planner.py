import numpy as np
import time
from fabrics.helpers.translation import c2np

"""
Use the example point_robot_urdf.py to generate a simple planner
```python
planner.export_as_c(simple_planner.c')
```

Then you can use this example to test the numpy function for parallelization.
"""

#python_code = c2np('planner.c', 'planner.py' )
from planner import casadi_f0_numpy

N = 300

q0 = np.array([[-1.9998125,0.5, 0, 0, 0, 0, 0]])
qdot0 = np.array([[0.0187499776, 0.0, 0, 0, 0, 0, 0]])
x_goal = np.array([[3.5, 0.5, 0.1]])

weight_goal_0_n = np.repeat(np.array([0.5]), N)
weight_goal_1_n = np.repeat(np.array([0.5]), N)
radius_obst_0_n = np.repeat(np.array([1.0]), N)
radius_obst_1_n = np.repeat(np.array([1.0]), N)
radius_body_n = np.repeat(np.array([0.2]), N)

q = np.repeat(np.transpose(q0), N, axis=1)
qdot = np.repeat(np.transpose(qdot0), N, axis=1)
x_obst_0 = np.array([[2.0, 0.0, 0.0]])
x_obst_0_n = np.repeat(np.transpose(x_obst_0), N, axis=1)
x_obst_1 = np.array([[2.0, 0.0, 0.0]])
x_obst_1_n = np.repeat(np.transpose(x_obst_1), N, axis=1)
constraint = np.array([[0, 0, 1, 0.0]])
constraint_n = np.repeat(np.transpose(constraint), N, axis=1)

x_goal_0_n = np.repeat(np.transpose(x_goal), N, axis=1)
x_goal_1_n = np.repeat(np.transpose(x_goal), N, axis=1)
# compute in in numpy parallel
t0 = time.perf_counter()
res2 = casadi_f0_numpy(
        constraint_n,
        q,
        qdot,
        radius_body_n,
        radius_body_n,
        radius_body_n,
        radius_body_n,
        radius_body_n,
        radius_obst_0_n,
        weight_goal_0_n,
        radius_obst_1_n,
        weight_goal_1_n,
        x_goal_0_n, 
        x_obst_0_n,
        x_goal_1_n,
        x_obst_1_n
)
t1 = time.perf_counter()
time_parallel = t1-t0

t0 = time.perf_counter()

# compute in loop
for i in range(N):
    res = casadi_f0_numpy(
            constraint_n[:,i:i+1],
            q[:,i:i+1],
            qdot[:,i:i+1],
            radius_body_n[i:i+1],
            radius_body_n[i:i+1],
            radius_body_n[i:i+1],
            radius_body_n[i:i+1],
            radius_body_n[i:i+1],
            radius_obst_0_n[i:i+1],
            weight_goal_0_n[i:i+1],
            radius_obst_1_n[i:i+1],
            weight_goal_1_n[i:i+1],
            x_goal_0_n[:,i:i+1],
            x_obst_0_n[:,i:i+1],
            x_goal_1_n[:,i:i+1],
            x_obst_1_n[:,i:i+1]
        )
    #print(res)
t1 = time.perf_counter()
time_loop = t1-t0
print(f'Parallel: {time_parallel} Loop: {time_loop} Ratio: {time_loop/time_parallel}')




