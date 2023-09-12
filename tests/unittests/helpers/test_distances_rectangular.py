import time
import casadi as ca
import pytest
import numpy as np
from fabrics.helpers.distances import (
    rectangle_to_point,
    rectangle_to_line,
    cuboid_to_sphere,
    cuboid_to_capsule,
)


def test_distance_rectangle_point():
    rectangle_center = ca.SX.sym("rectangle_center", 2)
    rectangle_size = ca.SX.sym("rectangle_size", 2)
    point = ca.SX.sym("point", 2)
    distance_expression = rectangle_to_point(rectangle_center, rectangle_size, point)
    function_arguments = [rectangle_center, rectangle_size, point]
    distance_function = ca.Function(
        "distance_function", function_arguments, [distance_expression]
    )

    rectangle_center_numpy = np.array([0.0, 0.0])
    rectangle_size_numpy = np.array([1.0, 1.0])
    point_numpy = np.array([3.0, 0.0])
    distance_numpy = np.array(
        distance_function(rectangle_center_numpy, rectangle_size_numpy, point_numpy)
    )
    assert distance_numpy == 2.5
    point_numpy = np.array([3.0, 2.0])
    distance_numpy = np.array(
        distance_function(rectangle_center_numpy, rectangle_size_numpy, point_numpy)
    )
    assert distance_numpy == (2.5**2+1.5**2)**0.5

def test_distance_rectangle_line():
    rectangle_center = ca.SX.sym("rectangle_center", 2)
    rectangle_size = ca.SX.sym("rectangle_size", 2)
    line_start = ca.SX.sym("line_start", 2)
    line_end = ca.SX.sym("line_end", 2)
    distance_expression = rectangle_to_line(
        rectangle_center,
        rectangle_size,
        line_start,
        line_end
    )
    function_arguments = [rectangle_center, rectangle_size, line_start, line_end]
    distance_function = ca.Function(
        "distance_function", function_arguments, [distance_expression]
    )

    rectangle_center_numpy = np.array([0.0, 0.0])
    rectangle_size_numpy = np.array([1.0, 1.0])
    line_start_numpy = np.array([3.0, 4.0])
    line_end_numpy = np.array([3.0, 0.0])
    distance_numpy = np.array(
        distance_function(
            rectangle_center_numpy,
            rectangle_size_numpy,
            line_start_numpy,
            line_end_numpy,
        )
    )
    assert distance_numpy == pytest.approx(2.5)
    line_start_numpy = np.array([1.5, 0.5])
    line_end_numpy = np.array([-0.5, -1.5])
    distance_numpy = np.array(
        distance_function(
            rectangle_center_numpy,
            rectangle_size_numpy,
            line_start_numpy,
            line_end_numpy,
        )
    )
    assert distance_numpy == pytest.approx(0.0)
    line_start_numpy = np.array([1.5, 2.5])
    line_end_numpy = np.array([1.5, -2.5])
    distance_numpy = np.array(
        distance_function(
            rectangle_center_numpy,
            rectangle_size_numpy,
            line_start_numpy,
            line_end_numpy,
        )
    )
    assert distance_numpy == pytest.approx(1.0)

def test_distance_cuboid_point():
    rectangle_center = ca.SX.sym("rectangle_center", 3)
    rectangle_size = ca.SX.sym("rectangle_size", 3)
    point = ca.SX.sym("point", 3)
    distance_expression = rectangle_to_point(rectangle_center, rectangle_size, point)
    function_arguments = [rectangle_center, rectangle_size, point]
    distance_function = ca.Function(
        "distance_function", function_arguments, [distance_expression]
    )
    rectangle_center_numpy = np.array([0.0, 0.0, 0.0])
    rectangle_size_numpy = np.array([1.0, 1.0, 0.0])

    point_numpy = np.array([3.0, 0.0, 0.0])
    distance_numpy = np.array(
        distance_function(rectangle_center_numpy, rectangle_size_numpy, point_numpy)
    )
    assert distance_numpy == 2.5
    rectangle_center_numpy = np.array([0.5, 0.5, 0.5])
    rectangle_size_numpy = np.array([1.0, 1.0, 1.0])

    point_numpy = np.array([1.5, 0.5, 0.0])
    distance_numpy = np.array(
        distance_function(rectangle_center_numpy, rectangle_size_numpy, point_numpy)
    )
    assert distance_numpy == 0.5
    point_numpy = np.array([0.5, 0.5, 1.0])
    distance_numpy = np.array(
        distance_function(rectangle_center_numpy, rectangle_size_numpy, point_numpy)
    )
    assert distance_numpy == 0.0

def test_distance_cuboid_sphere():
    cuboid_center = ca.SX.sym("cuboid_center", 3)
    cuboid_size = ca.SX.sym("cuboid_size", 3)
    sphere_center = ca.SX.sym("sphere_center", 3)
    sphere_size = ca.SX.sym("sphere_size", 1)
    distance_expression = cuboid_to_sphere(
        cuboid_center,
        sphere_center,
        cuboid_size,
        sphere_size
    )
    function_arguments = [cuboid_center, sphere_center, cuboid_size, sphere_size]
    distance_function = ca.Function(
        "distance_function", function_arguments, [distance_expression]
    )
    cuboid_center_numpy = np.array([0.0, 0.0, 0.0])
    cuboid_size_numpy = np.array([1.0, 1.0, 0.0])
    sphere_center_numpy = np.array([3.0, 0.0, 0.0])
    sphere_size_numpy = np.array([0.2])
    distance_numpy = np.array(
        distance_function(
            cuboid_center_numpy,
            sphere_center_numpy,
            cuboid_size_numpy,
            sphere_size_numpy)
    )
    assert distance_numpy == 2.3
    cuboid_center_numpy = np.array([0.5, 0.5, 0.5])
    cuboid_size_numpy = np.array([1.0, 1.0, 1.0])
    sphere_size_numpy = np.array([0.2])

    sphere_center_numpy = np.array([1.5, 0.5, 0.0])
    distance_numpy = np.array(
        distance_function(
            cuboid_center_numpy,
            sphere_center_numpy,
            cuboid_size_numpy,
            sphere_size_numpy)
    )
    assert distance_numpy == 0.3

    sphere_center_numpy = np.array([0.5, 0.2, 1.0])
    distance_numpy = np.array(
        distance_function(
            cuboid_center_numpy,
            sphere_center_numpy,
            cuboid_size_numpy,
            sphere_size_numpy)
    )
    assert distance_numpy == 0.0

def test_distance_cuboid_capsule():
    cuboid_center = ca.SX.sym("cuboid_center", 3)
    cuboid_size = ca.SX.sym("cuboid_size", 3)
    capsule_centers = [
        ca.SX.sym("capsule_center1", 3),
        ca.SX.sym("capsule_center2", 3),
    ]
    capsule_radius = ca.SX.sym("capsule_radius", 1)
    distance_expression = cuboid_to_capsule(
        cuboid_center,
        capsule_centers,
        cuboid_size,
        capsule_radius,
    )
    function_arguments = capsule_centers + [cuboid_center, cuboid_size, capsule_radius]
    distance_function = ca.Function(
        "distance_function", function_arguments, [distance_expression]
    )
    cuboid_center_numpy = np.array([0.5, 0.5, 0.5])
    cuboid_size_numpy = np.array([1.0, 1.0, 1.0])
    capsule_radius_numpy = np.array([0.2])

    capsule_centers_numpy = [
        np.array([-2.0, 3.0, 0.5]),
        np.array([2.0, 3.0, 0.5])
    ]
    t0 = time.perf_counter()
    distance_numpy = np.array(
        distance_function(
            capsule_centers_numpy[0],
            capsule_centers_numpy[1],
            cuboid_center_numpy,
            cuboid_size_numpy,
            capsule_radius_numpy,
        )
    )
    t1 = time.perf_counter()
    print(t1-t0)
    assert distance_numpy == 1.8

    capsule_radius_numpy = np.array([0.2])
    capsule_centers_numpy = [
        np.array([0.0, 3.0, 0.15]),
        np.array([3.0, 0.0, 0.75])
    ]
    distance_numpy = np.array(
        distance_function(
            capsule_centers_numpy[0],
            capsule_centers_numpy[1],
            cuboid_center_numpy,
            cuboid_size_numpy,
            capsule_radius_numpy,
        )
    )
    assert distance_numpy == pytest.approx(0.70710678-0.2)
