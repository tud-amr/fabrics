from planar_arm import run_planar_arm_example
from planar_point_robot import run_point_robot_example
from planar_point_robot_line_goal import run_point_robot_line_goal_example
from panda import run_panda_example
from panda_orientation import run_panda_orientation_example

def test_panda_example():
    res = run_panda_example(10, render=False)
    assert isinstance(res, dict)

def test_planar_arm_example():
    res = run_planar_arm_example(10, render=False)
    assert isinstance(res, dict)

def test_planar_point_robot_example():
    res = run_point_robot_example(10, render=False)
    assert isinstance(res, dict)

def test_planar_point_robot_line_goal_example():
    res = run_point_robot_line_goal_example(10, render=False)
    assert isinstance(res, dict)

def test_panda_orientation_example():
    res = run_panda_orientation_example(10, render=False)
    assert isinstance(res, dict)


