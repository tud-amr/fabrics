import casadi as ca
from typing import List


def point_to_line(point: ca.SX, line_start: ca.SX, line_end: ca.SX) -> ca.SX:
    line_vec = line_end - line_start
    point_vec = point - line_start
    proj_length = ca.dot(point_vec, line_vec) / ca.norm_2(line_vec)
    distance_0 = ca.norm_2(point - line_start)
    distance_1 = ca.norm_2(point - line_end)
    proj_point = line_start + proj_length * line_vec / ca.norm_2(line_vec)
    distance_2 = ca.norm_2(point - proj_point)

    distance = ca.if_else(
        proj_length <= 0,
        distance_0,
        ca.if_else(
            proj_length >= ca.norm_2(line_vec),
            distance_1,
            distance_2,
        ),
    )
    return distance

def line_to_line(line_1_start: ca.SX, line_1_end: ca.SX, line_2_start: ca.SX, line_2_end: ca.SX) -> ca.SX:
    """
    Computes the distance between two lines ignoring the case that they are 
    intersecting.
    """
    distance = ca.fmin(
        point_to_line(line_1_start, line_2_start, line_2_end), 
        point_to_line(line_1_end, line_2_start, line_2_end), 
    )
    return distance


def point_to_plane(point: ca.SX, plane: ca.SX) -> ca.SX:
    distance = ca.fabs(ca.dot(plane[0:3], point) + plane[3]) / ca.norm_2(
        plane[0:3]
    )
    return distance


def sphere_to_plane(
    sphere_center: ca.SX, plane: ca.SX, sphere_radius: ca.SX
) -> ca.SX:
    distance = point_to_plane(sphere_center, plane) - sphere_radius
    return distance


def line_to_plane(line_start: ca.SX, line_end: ca.SX, plane: ca.SX):
    distance_line_start = point_to_plane(line_start, plane)
    distance_line_end = point_to_plane(line_end, plane)
    min_distance_ends = ca.fmin(distance_line_start, distance_line_end)
    product_dot_products = ca.dot(plane[0:3], line_start) * ca.dot(
        plane[0:3], line_end
    )
    distance = ca.if_else(product_dot_products < 0, 0.0, min_distance_ends)
    return distance

def capsule_to_plane(
    capsule_centers: List[ca.SX],
    plane: ca.SX,
    capsule_radius: ca.SX,
) -> ca.SX:
    return line_to_plane(capsule_centers[0], capsule_centers[1], plane) - capsule_radius

def capsule_to_capsule(
    capsule_1_centers: List[ca.SX],
    capsule_2_centers: List[ca.SX],
    capsule_1_radius: ca.SX,
    capsule_2_radius: ca.SX,
) -> ca.SX:
    return line_to_line(
        capsule_1_centers[0], capsule_1_centers[1],
        capsule_2_centers[0], capsule_2_centers[1]
    ) - capsule_1_radius - capsule_2_radius


def capsule_to_sphere(
    capsule_centers: List[ca.SX],
    sphere_center: ca.SX,
    capsule_radius: ca.SX,
    sphere_radius: ca.SX,
) -> ca.SX:
    assert len(capsule_centers) == 2
    distance_line_center = point_to_line(
        sphere_center, capsule_centers[0], capsule_centers[1]
    )
    return distance_line_center - capsule_radius - sphere_radius
