
def test_panda_example():
    from panda import run_panda_example
    res = run_panda_example(10, render=False)
    assert isinstance(res, dict)

def test_panda_ring_example():
    from panda_ring import run_panda_ring_example
    res = run_panda_ring_example(n_steps=10, render=False, serialize=True)
    assert isinstance(res, dict)

def test_panda_trajectory():
    from panda_trajectory_following import run_panda_trajectory_example
    res = run_panda_trajectory_example(10, render=False)
    assert isinstance(res, dict)

def test_panda_self_collision_avoidance():
    from panda_self_collision_avoidance import run_panda_self_collision
    res = run_panda_self_collision(10, render=False)
    assert isinstance(res, dict)

"""
def test_planar_arm_example():
    from planar_arm import run_planar_arm_example
    res = run_planar_arm_example(10, render=False)
    assert isinstance(res, dict)

def test_planar_arm_limits_example():
    from planar_arm_limits import run_planar_arm_limits_example
    res = run_planar_arm_limits_example(10, render=False)
    assert isinstance(res, dict)

def test_planar_point_robot_example():
    from planar_point_robot import run_point_robot_example
    res = run_point_robot_example(10, render=False)
    assert isinstance(res, dict)

def test_planar_point_symbolic():
    from planar_point_robot_symbolic import run_point_robot_symbolic
    res = run_point_robot_symbolic(n_steps=10, render=False)
    assert isinstance(res, dict)

def test_planar_point_robot_line_goal_example():
    from planar_point_robot_line_goal import run_point_robot_line_goal_example
    res = run_point_robot_line_goal_example(10, render=False)
    assert isinstance(res, dict)
"""

def test_serialization_example():
    from panda_ring_serialized import run_panda_ring_serialized_example
    from panda_ring import run_panda_ring_example
    run_panda_ring_example(n_steps=10, render=False, serialize=True)
    res = run_panda_ring_serialized_example(n_steps=10, render=False)
    assert isinstance(res, dict)

def test_panda_joint_space():
    from panda_joint_space import run_panda_joint_space
    res = run_panda_joint_space(10, render=False)
    assert isinstance(res, dict)

def test_point_robot_urdf_example():
    from point_robot_urdf import run_point_robot_urdf
    res = run_point_robot_urdf(10, render=False)
    assert isinstance(res, dict)

def test_point_robot_urdf_passage_example():
    from point_robot_urdf_passage import run_point_robot_urdf
    res = run_point_robot_urdf(10, render=False)
    assert isinstance(res, dict)

def test_tiago_example():
    from tiago_arm import run_tiago_example
    res = run_tiago_example(10, render=False)
    assert isinstance(res, dict)

def test_boxer_example():
    from boxer import run_boxer_example
    res = run_boxer_example(10, render=False)
    assert isinstance(res, dict)

def test_albert_example():
    from albert import run_albert_reacher_example
    res = run_albert_reacher_example(10, render=False)
    assert isinstance(res, dict)

def test_esdf_point_robot():
    from point_robot_esdf import run_point_robot_esdf
    res = run_point_robot_esdf(10, render=False)
    assert isinstance(res, dict)

def test_esdf_planar_robot():
    from planar_robot_esdf import run_planar_robot_esdf
    res = run_planar_robot_esdf(10, render=False)
    assert isinstance(res, dict)

