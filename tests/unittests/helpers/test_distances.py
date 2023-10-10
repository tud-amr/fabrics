import casadi as ca
import pytest
import numpy as np
from fabrics.helpers.distances import *

def test_closest_point_line():
    line_start = ca.SX.sym("line_start", 3)
    line_end = ca.SX.sym("line_end", 3)
    point = ca.SX.sym("point", 3)
    closest_point_expression = closest_point_to_line(point, line_start, line_end)
    function_arguments = [point, line_start, line_end]
    closest_point_function = ca.Function(
        "distance_function", function_arguments, [closest_point_expression]
    )
    line_start_numpy = np.array([1, 2, 0])
    line_end_numpy = np.array([1, -1, 0])
    point_numpy = np.array([3.0, 1.2, 0.0])
    closest_point_numpy = np.array(
        closest_point_function(point_numpy, line_start_numpy, line_end_numpy)
    )[:, 0]
    assert closest_point_numpy[0] == pytest.approx(1.0)
    assert closest_point_numpy[1] == pytest.approx(1.2)
    assert closest_point_numpy[2] == pytest.approx(0.0)

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

def test_distance_line_line():
    line_1_start = ca.SX.sym("line_1_start", 3)
    line_1_end = ca.SX.sym("line_1_end", 3)
    line_2_start = ca.SX.sym("line_2_start", 3)
    line_2_end = ca.SX.sym("line_2_end", 3)
    distance_expression = line_to_line(
        line_1_start,
        line_1_end,
        line_2_start,
        line_2_end,
    )
    function_arguments = [line_1_start, line_1_end, line_2_start, line_2_end]
    distance_function = ca.Function(
        "distance_function", function_arguments, [distance_expression]
    )

    line_1_start_numpy = np.array([1, -0.2, 0])
    line_1_end_numpy = np.array([-1, -0.2, 0])
    line_2_start_numpy = np.array([0, 1, 3])
    line_2_end_numpy = np.array([0, 1, -3])
    distance_numpy = np.array(
        distance_function(
            line_1_start_numpy,
            line_1_end_numpy,
            line_2_start_numpy,
            line_2_end_numpy,
        )
    )
    assert distance_numpy == 1.2

    line_1_start_numpy = np.array([1, -0.2, 0])
    line_1_end_numpy = np.array([-1, -0.2, 0])
    line_2_start_numpy = np.array([2, 1, 0])
    line_2_end_numpy = np.array([-2, 1, 0])
    distance_numpy = np.array(
        distance_function(
            line_1_start_numpy,
            line_1_end_numpy,
            line_2_start_numpy,
            line_2_end_numpy,
        )
    )
    assert distance_numpy == 1.2

    line_1_start_numpy = np.array([0, 0, 0])
    line_1_end_numpy = np.array([0, 1, 0])
    line_2_start_numpy = np.array([2, 2, 0])
    line_2_end_numpy = np.array([-2, 2, 0])
    distance_numpy = np.array(
        distance_function(
            line_1_start_numpy,
            line_1_end_numpy,
            line_2_start_numpy,
            line_2_end_numpy,
        )
    )
    assert distance_numpy == 1.0
    line_1_start_numpy = np.array([0, -1, 0])
    line_1_end_numpy = np.array([0, 1, 0])
    line_2_start_numpy = np.array([-2, 0, 0])
    line_2_end_numpy = np.array([-1, 0, 0])
    distance_numpy = np.array(
        distance_function(
            line_1_start_numpy,
            line_1_end_numpy,
            line_2_start_numpy,
            line_2_end_numpy,
        )
    )
    assert distance_numpy == 1.0
    line_1_start_numpy = np.array([-2.0, 1.5, 0])
    line_1_end_numpy = np.array([-2.0, 3.5, 0])
    line_2_start_numpy = np.array([0, -5, 0])
    line_2_end_numpy = np.array([0, 5, 0])
    distance_numpy = np.array(
        distance_function(
            line_1_start_numpy,
            line_1_end_numpy,
            line_2_start_numpy,
            line_2_end_numpy,
        )
    )
    assert distance_numpy == 2.0
    edges = [
        [[0, 0, 0], [1, 0, 0]],
        [[0, 0, 0], [0, 1, 0]],
        [[0, 0, 0], [0, 0, 1]],
        [[0, 0, 1], [1, 0, 1]],
        [[0, 0, 1], [0, 1, 1]],
        [[0, 1, 0], [1, 1, 0]],
        [[0, 1, 0], [0, 1, 1]],
        [[1, 0, 0], [1, 1, 0]],
        [[1, 0, 0], [1, 0, 1]],
        [[1, 1, 1], [0, 1, 1]],
        [[1, 1, 1], [1, 0, 1]],
        [[1, 1, 1], [1, 1, 0]],
    ]
    for edge in edges:
        line_1_start_numpy = np.array(edge[0])
        line_1_end_numpy = np.array(edge[1])
        line_2_start_numpy = np.array([-2, 3, 0.5])
        line_2_end_numpy = np.array([2, 3, 0.5])
        distance_numpy = np.array(
            distance_function(
                line_1_start_numpy,
                line_1_end_numpy,
                line_2_start_numpy,
                line_2_end_numpy,
            )
        )
        assert distance_numpy >= 2.0




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

    capsule_centers_numpy = [
        np.array([-1.0,  1.0, 0.0]),
        np.array([ 1.0, -1.0, 0.0]),
    ]
    sphere_center_numpy = np.array([1.0, 1.0, 0.0])
    sphere_radius_numpy = 0.0
    capsule_radius_numpy = 0.0
    distance_numpy = np.array(
        distance_function(
            capsule_centers_numpy[0],
            capsule_centers_numpy[1],
            sphere_center_numpy,
            capsule_radius_numpy,
            sphere_radius_numpy,
        )
    )
    assert distance_numpy == pytest.approx(np.sqrt(2.0) - 0.0 - 0.0)

def test_edges_of_cuboid():
    cuboid_center = ca.SX.sym("cuboid_center", 3)
    cuboid_size = ca.SX.sym("cuboid_size", 3)
    cuboid_center_numpy = np.array([1.0, 0.0, 0.3])
    cuboid_size_numpy = np.array([1.0, 0.5, 1.0])
    function_arguments = [cuboid_center, cuboid_size]

    edge_expression = edge_of_cuboid(cuboid_center, cuboid_size, 0)
    edge_function = ca.Function('edge_function', function_arguments, [edge_expression])

    edge_numpy = np.array(edge_function(cuboid_center_numpy, cuboid_size_numpy))[:, 0]
    assert np.array_equal(edge_numpy[0:3], np.array([0.5, -0.25, -0.2]))
    assert np.array_equal(edge_numpy[3:6], np.array([1.5, -0.25, -0.2]))

    edge_expression = edge_of_cuboid(cuboid_center, cuboid_size, 1)
    edge_function = ca.Function('edge_function', function_arguments, [edge_expression])

    edge_numpy = np.array(edge_function(cuboid_center_numpy, cuboid_size_numpy))[:, 0]
    assert np.array_equal(edge_numpy[0:3], np.array([0.5, -0.25, -0.2]))
    assert np.array_equal(edge_numpy[3:6], np.array([0.5, 0.25, -0.2]))


def test_distance_rectangle_point():
    """
    Compute distance between rectangle (2D) and a point (2D)
    """
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
    """
    Computes distance between rectangle (2D!) and line (2D)
    """
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
    """
    Test distance of a cuboid (3D) to a point (3D)
    """
    cuboid_center = ca.SX.sym("cuboid_center", 3)
    cuboid_size = ca.SX.sym("cuboid_size", 3)
    point = ca.SX.sym("point", 3)
    distance_expression = cuboid_to_point(cuboid_center, cuboid_size, point)
    function_arguments = [cuboid_center, cuboid_size, point]
    distance_function = ca.Function(
        "distance_function", function_arguments, [distance_expression]
    )
    cuboid_center_numpy = np.array([0.0, 0.0, 0.0])
    cuboid_size_numpy = np.array([1.0, 1.0, 0.0])

    point_numpy = np.array([3.0, 0.0, 0.0])
    distance_numpy = np.array(
        distance_function(cuboid_center_numpy, cuboid_size_numpy, point_numpy)
    )
    assert distance_numpy == 2.5
    cuboid_center_numpy = np.array([0.5, 0.5, 0.5])
    cuboid_size_numpy = np.array([1.0, 1.0, 1.0])

    point_numpy = np.array([1.5, 0.5, 0.0])
    distance_numpy = np.array(
        distance_function(cuboid_center_numpy, cuboid_size_numpy, point_numpy)
    )
    assert distance_numpy == 0.5
    point_numpy = np.array([0.5, 0.5, 1.0])
    distance_numpy = np.array(
        distance_function(cuboid_center_numpy, cuboid_size_numpy, point_numpy)
    )
    assert distance_numpy == 0.0

def test_distance_cuboid_sphere():
    """
    Test distance of a cuboid (3D) to a sphere (3D)
    """
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
    """
    Test distance of a cuboid (3D) to a capsule (3D)
    """
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
    distance_numpy = np.array(
        distance_function(
            capsule_centers_numpy[0],
            capsule_centers_numpy[1],
            cuboid_center_numpy,
            cuboid_size_numpy,
            capsule_radius_numpy,
        )
    )
    assert distance_numpy == 1.8
    
    cuboid_center_numpy = np.array([0.0, 0.0, 0.0])
    cuboid_size_numpy = np.array([1.0, 1.0, 1.0])
    capsule_radius_numpy = np.array([0.0])

    capsule_centers_numpy = [
        np.array([1.5, 0.5, 0.5]),
        np.array([-0.5, -1.5, 0.5])
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
    assert distance_numpy == 0.0

    cuboid_center_numpy = np.array([0.5, 0.5, 0.5])
    cuboid_size_numpy = np.array([1.0, 1.0, 1.0])
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
