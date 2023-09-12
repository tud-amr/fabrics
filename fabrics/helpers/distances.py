import casadi as ca
import numpy as np
from typing import List, Union

def closest_point_to_line(point: ca.SX, line_start: ca.SX, line_end: ca.SX) -> ca.SX:
    line_vector = line_end - line_start
    point_vector = point - line_start
    t = ca.dot(point_vector, line_vector) / ca.dot(line_vector, line_vector)
    t = ca.fmax(0, ca.fmin(1, t))
    return line_start + t * line_vector

def clamp(a: ca.SX, a_min: float, a_max: float):
    return ca.fmin(a_max, ca.fmax(a, a_min))



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

def line_to_line_sampled(
    line_1_start: ca.SX,
    line_1_end: ca.SX,
    line_2_start: ca.SX,
    line_2_end: ca.SX,
    samples: int = 100) -> ca.SX:
    """
    Computes the distance between two lines ignoring the case that they are 
    intersecting.
    """
    distance = ca.fmin(
        point_to_line(line_1_start, line_2_start, line_2_end), 
        point_to_line(line_1_end, line_2_start, line_2_end), 
    )
    for i in range(samples):
        eta = 1/samples * i
        line_point = (1-eta) * line_1_start + eta * line_1_end
        distance = ca.fmin(distance, point_to_line(line_point, line_2_start, line_2_end))

    return distance

def line_to_line(
    line_1_start: ca.SX,
    line_1_end: ca.SX,
    line_2_start: ca.SX,
    line_2_end: ca.SX) -> ca.SX:
    """
    Computes the distance between two lines according to 
    Real-Time Collision Detection by Christer Ericson, page 148
    """
    eps = 1e-5
    d1 = line_1_end - line_1_start
    d2 = line_2_end - line_2_start
    r = line_1_start - line_2_start
    a = ca.dot(d1, d1)
    e = ca.dot(d2, d2)
    f = ca.dot(d2, r)
    c = ca.dot(d1, r)
    b = ca.dot(d1, d2)
    denom = a*e-b*b
    s = ca.if_else(
        a <= eps,
        0.0,
        ca.if_else(
            e <= eps,
            clamp(-c/a, 0.0, f),
            ca.if_else(
                denom != 0.0,
                clamp((b*f- c*e)/denom, 0.0, f),
                0.0
            )
        )
    )

    t = ca.if_else(
        a <= eps,
        clamp(f / e, 0.0, f),
        ca.if_else(
            e <= eps,
            0.0,
            (b*s+f)/e
        )
    )
    s_1 = ca.if_else(
        t < 0.0,
        clamp(-c / a, 0.0, f),
        s
    )
    s_2 = ca.if_else(
        t > f,
        clamp((b*f-c*e)/denom, 0.0, f),
        s_1
    )
    t_1 = clamp(t, 0.0, f)
    c1 = line_1_start + d1 * s_2
    c2 = line_2_start + d2 * t_1
    distance = ca.if_else(
        ca.logic_and(a <= eps, e<=eps),
        ca.sqrt(ca.dot(line_1_start - line_2_start, line_1_start - line_2_start)),
        ca.dot(c1-c2, c1-c2)
    )
    return ca.sqrt(distance)

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
    return ca.fmax(distance_line_center - capsule_radius - sphere_radius, 0.0)


def cuboid_to_point_half_distances(
        rectangle_center: ca.SX,
        rectangle_size: ca.SX,
        point: ca.SX,
    ) -> List[ca.SX]:
    half_distances = []
    for i in range(point.size()[0]):
        half_distances.append(
            ca.fmax(ca.fabs(point[i] - rectangle_center[i]) - rectangle_size[i]/2, 0.0))
    return half_distances

def rectangle_to_point(
        rectangle_center: ca.SX,
        rectangle_size: ca.SX,
        point: ca.SX,
    ) -> ca.SX:
    half_distances = cuboid_to_point_half_distances(
        rectangle_center,
        rectangle_size,
        point
    )
    return ca.sqrt(half_distances[0] ** 2 + half_distances[1] ** 2)

def rectangle_to_line(
        rectangle_center: ca.SX,
        rectangle_size: ca.SX,
        line_start: ca.SX,
        line_end: ca.SX,
    ) -> ca.SX:
    min_distance = ca.fmin(
        rectangle_to_point(rectangle_center, rectangle_size, line_start), 
        rectangle_to_point(rectangle_center, rectangle_size, line_end), 
    )
    for i in [-1, 1]:
        for j in [-1, 1]:
            index = [i, j]
            corner_transform = rectangle_size / 2 * index
            corner = rectangle_center + corner_transform
            min_distance = ca.fmin(min_distance, point_to_line(corner, line_start, line_end))
    return min_distance

def cuboid_to_point(
        cuboid_center: ca.SX,
        cuboid_size: ca.SX,
        point: ca.SX,
    ) -> ca.SX:
    half_distances = cuboid_to_point_half_distances(
        cuboid_center,
        cuboid_size,
        point
    )
    return ca.sqrt(
        half_distances[0] ** 2 + half_distances[1] ** 2 + half_distances[2] ** 2
    )

def cuboid_to_line(
        cuboid_center: ca.SX,
        cuboid_size: ca.SX,
        line_start: ca.SX,
        line_end: ca.SX,
        samples: int = 100,
    ) -> ca.SX:
    distance = ca.SX(ca.inf)
    for i in range(samples+1):
        eta = 1/samples * i
        line_point = (1-eta) * line_start + eta * line_end
        distance = ca.fmin(distance, cuboid_to_point(cuboid_center, cuboid_size, line_point))
    return distance

def cuboid_to_sphere(
        cuboid_center: ca.SX,
        sphere_center: ca.SX,
        cuboid_size: ca.SX,
        sphere_size: ca.SX,
    ) -> ca.SX:
    return ca.fmax(0.0, cuboid_to_point(cuboid_center, cuboid_size, sphere_center) - sphere_size)

def cuboid_to_capsule(
        cuboid_center: ca.SX,
        capsule_centers: List[ca.SX],
        cuboid_size: ca.SX,
        capsule_radius: ca.SX,
    ) -> ca.SX:
    return ca.fmax(cuboid_to_line(cuboid_center, cuboid_size, capsule_centers[0], capsule_centers[1]) - capsule_radius, 0.0)
