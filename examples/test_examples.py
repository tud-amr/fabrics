from planar_arm import run_planar_arm_example
from planar_arm_limits import run_planar_arm_limits_example
from planar_point_robot import run_point_robot_example
from planar_point_robot_symbolic import run_point_robot_symbolic
from planar_point_robot_line_goal import run_point_robot_line_goal_example
from panda import run_panda_example
from panda_orientation import run_panda_orientation_example
from panda_trajectory_following import run_panda_trajectory_example
from panda_ring import run_panda_ring_example
from panda_ring_serialized import run_panda_ring_serialized_example
from panda_self_collision_avoidance import run_panda_self_collision
from panda_joint_space import run_panda_joint_space
from point_robot_urdf import run_point_robot_urdf
from tiago_arm import run_tiago_example

def test_panda_example():
    res = run_panda_example(10, render=False)
    assert isinstance(res, dict)

def test_panda_ring_example():
    res = run_panda_ring_example(n_steps=10, render=False, serialize=True)
    assert isinstance(res, dict)

def test_panda_trajectory():
    res = run_panda_trajectory_example(10, render=False)
    assert isinstance(res, dict)

def test_panda_self_collision_avoidance():
    res = run_panda_self_collision(10, render=False)
    assert isinstance(res, dict)

def test_planar_arm_example():
    res = run_planar_arm_example(10, render=False)
    assert isinstance(res, dict)

def test_planar_arm_limits_example():
    res = run_planar_arm_limits_example(10, render=False)
    assert isinstance(res, dict)

def test_planar_point_robot_example():
    res = run_point_robot_example(10, render=False)
    assert isinstance(res, dict)

def test_planar_point_symbolic():
    res = run_point_robot_symbolic(n_steps=10, render=False)
    assert isinstance(res, dict)

def test_planar_point_robot_line_goal_example():
    res = run_point_robot_line_goal_example(10, render=False)
    assert isinstance(res, dict)

def test_panda_orientation_example():
    res = run_panda_orientation_example(10, render=False)
    assert isinstance(res, dict)

def test_serialization_example():
    run_panda_ring_example(n_steps=10, render=False, serialize=True)
    res = run_panda_ring_serialized_example(n_steps=10, render=False)
    assert isinstance(res, dict)

def test_panda_joint_space():
    res = run_panda_joint_space(10, render=False)
    assert isinstance(res, dict)

def test_point_robot_urdf_example():
    res = run_point_robot_urdf(10, render=False)
    assert isinstance(res, dict)

def test_tiago_example():
    res = run_tiago_example(10, render=False)
    assert isinstance(res, dict)


