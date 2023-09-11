import casadi as ca
import pytest
import numpy as np
from fabrics.helpers.distances import (
    point_to_line,
    point_to_rectangle,
    rectangle_struct,
    sphere_to_rectangle,
    capsule_to_rectangle,
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
def test_distance_point_rectangle():
    # Center and size (length, width, height) of rectangle
    center = np.array([0.5, 0.5, 0.5])
    size = np.array([1, 1, 1])

    #Points to test:
    p_list = [np.array([1.5, 0.5, 0.0]), np.array([0.5, 0.0, 1.0])]
    rect = rectangle_struct(center, size)
    for index, p in enumerate(p_list):
        dist = point_to_rectangle(p, rect)
        if index == 0:
            dist_hardcode = 0.5
            assert dist_hardcode == dist
        elif index == 1:
            dist_hardcode = 0.0
            assert dist_hardcode == dist
        else:
            print("warning, test 'rectangle to point' failed")
        print("distance:", dist)

def test_distance_sphere_rectangle():
    # Center and size (length, width, height) of rectangle
    center = np.array([0.5, 0.5, 0.5])
    size = np.array([1, 1, 1])

    #Spheres to test:
    p_list = [np.array([1.5, 0.5, 0.0]), np.array([0.5, 0.2, 1.0])]
    sphere_radius = 0.2
    rect = rectangle_struct(center, size)
    for index, p in enumerate(p_list):
        dist = sphere_to_rectangle(p, rect, sphere_radius)
        if index == 0:
            dist_hardcode = 0.3
            assert dist_hardcode == dist
        elif index == 1:
            dist_hardcode = 0.0
            assert dist_hardcode == dist
        else:
            print("warning, test 'rectangle to point' failed")
        print("distance:", dist)

def test_distance_point_capsule():
    # Center and size (length, width, height) of rectangle
    center = np.array([0.5, 0.5, 0.5])
    size = np.array([1, 1, 1])

    #Spheres to test:
    capsule_centers_list = [[np.array([-2.0, 3.0, 0.5]), np.array([2.0, 3.0, 0.5])],
               [np.array([-2.0, 3.0, 0.5]), np.array([2.0, 3.0, 0.5])]]
    capsule_radius = 0.2
    rect = rectangle_struct(center, size)
    for index, capsule_centers in enumerate(capsule_centers_list):
        dist = capsule_to_rectangle(capsule_centers, rect, capsule_radius)
        if index == 0:
            dist_hardcode = 1.8
            assert dist_hardcode == dist
        elif index == 1:
            dist_hardcode = 1.8
            assert dist_hardcode == dist
        else:
            print("warning, test 'rectangle to point' failed")
        print("distance:", dist)




