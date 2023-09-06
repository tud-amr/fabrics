import casadi as ca
import pytest
import numpy as np
from fabrics.helpers.distances import (
    point_to_line,
    point_to_plane,
    line_to_plane,
    capsule_to_sphere,
)


def test_distance_point_line():
    line_start = ca.SX.sym("line_start", 3)
    line_end = ca.SX.sym("line_end", 3)
    point = ca.SX.sym("point", 3)
    distance_expression = point_to_line(point, line_start, line_end)
    function_arguments = [point, line_start, line_end]
    distance_function = ca.Function(
        "distance_function", function_arguments, [distance_expression]
    )

    line_start_numpy = np.array([1, 1, 0])
    line_end_numpy = np.array([2, 2, 0])
    point_numpy = np.array([3.0, 1.0, 0.0])
    distance_numpy = np.array(
        distance_function(point_numpy, line_start_numpy, line_end_numpy)
    )
    assert distance_numpy == np.sqrt(2)

    point_numpy = np.array([3.0, 2.0, 0.0])
    distance_numpy = np.array(
        distance_function(point_numpy, line_start_numpy, line_end_numpy)
    )
    assert distance_numpy == 1.0

    point_numpy = np.array([0.0, 0.0, 0.0])
    distance_numpy = np.array(
        distance_function(point_numpy, line_start_numpy, line_end_numpy)
    )
    assert distance_numpy == np.sqrt(2)


def test_distance_point_plane():
    plane = ca.SX.sym("plane", 4)
    point = ca.SX.sym("point", 3)
    distance_expression = point_to_plane(point, plane)
    function_arguments = [point, plane]
    distance_function = ca.Function(
        "distance_function", function_arguments, [distance_expression]
    )

    plane_numpy = np.array([2, 1, -1, 3])
    point_numpy = np.array([2.0, 1.1, -0.3])
    distance_numpy = np.array(distance_function(point_numpy, plane_numpy))
    assert distance_numpy == pytest.approx(3.429285639896449)

    plane_numpy = np.array([-2, -1, 0, 0])
    point_numpy = np.array([1.0, 2.0, 0.0])
    distance_numpy = np.array(distance_function(point_numpy, plane_numpy))
    print(distance_numpy)
    assert distance_numpy == pytest.approx(1.788854381999832)


def test_distance_line_plane():
    plane = ca.SX.sym("plane", 4)
    line_start = ca.SX.sym("line_start", 3)
    line_end = ca.SX.sym("line_end", 3)
    distance_expression = line_to_plane(line_start, line_end, plane)
    function_arguments = [line_start, line_end, plane]
    distance_function = ca.Function(
        "distance_function", function_arguments, [distance_expression]
    )

    plane_numpy = np.array([-2, -1, 0, 0])
    line_start_numpy = np.array([1.0, 0.0, 0.0])
    line_end_numpy = np.array([5.0, 1.1, 0.0])
    distance_numpy = np.array(
        distance_function(line_start_numpy, line_end_numpy, plane_numpy)
    )
    cos_arctan = np.cos(np.arctan(0.5))
    assert distance_numpy == pytest.approx(cos_arctan)

    plane_numpy = np.array([-2, -1, 0, 0])
    line_start_numpy = np.array([1.0, 0.0, 0.0])
    line_end_numpy = np.array([-5.0, 1.1, 0.0])
    distance_numpy = np.array(
        distance_function(line_start_numpy, line_end_numpy, plane_numpy)
    )
    assert distance_numpy == pytest.approx(0.0)


def test_distance_capsule_sphere():
    capsule_centers = [
        ca.SX.sym("capsule_center1", 3),
        ca.SX.sym("capsule_center2", 3),
    ]
    sphere_center = ca.SX.sym("sphere_center", 3)
    capsule_radius = ca.SX.sym("capsule_radius")
    sphere_radius = ca.SX.sym("sphere_radius")
    distance_expression = capsule_to_sphere(
        capsule_centers, sphere_center, capsule_radius, sphere_radius
    )
    function_arguments = capsule_centers + [
        sphere_center,
        capsule_radius,
        sphere_radius,
    ]
    distance_function = ca.Function(
        "distance_function", function_arguments, [distance_expression]
    )
    capsule_centers_numpy = [
        np.array([1, 1, 0]),
        np.array([2, 2, 0]),
    ]
    sphere_center_numpy = np.array([3.0, 1.0, 0.0])
    sphere_radius_numpy = 0.4
    capsule_radius_numpy = 0.3
    distance_numpy = np.array(
        distance_function(
            capsule_centers_numpy[0],
            capsule_centers_numpy[1],
            sphere_center_numpy,
            capsule_radius_numpy,
            sphere_radius_numpy,
        )
    )
    assert distance_numpy == np.sqrt(2) - 0.3 - 0.4
