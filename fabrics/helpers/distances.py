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

def compute_rectangle_edges(
        center: ca.SX,
        size: ca.SX):
    half_length = size[0] / 2
    half_width = size[1] / 2
    half_height = size[2] / 2

    # Define the equations for the six planes of the rectangle
    # Define the corners of the rectangle relative to the center
    corners = []
    for i_x in range(2):
        dx = (2*i_x - 1) * size[0] / 2
        for i_y in range(2):
            dy = (2 * i_y - 1) * size[1] / 2
            for i_z in range(2):
                dz = (2 * i_z - 1) * size[2] / 2
                corner = [center[0]+dx, center[1]+dy, center[2]+dz]
                corners.append(corner)

    # corner_lines = [
    #     [center[0] - half_length, center[1] + half_width, center[2] - half_height],
    #     [center[0] + half_length, center[1] + half_width, center[2] - half_height],
    #     [center[0] - half_length, center[1] - half_width, center[2] - half_height],
    #     [center[0] + half_length, center[1] - half_width, center[2] - half_height],
    #     [center[0] - half_length, center[1] + half_width, center[2] + half_height],
    #     [center[0] + half_length, center[1] + half_width, center[2] + half_height],
    #     [center[0] - half_length, center[1] - half_width, center[2] + half_height],
    #     [center[0] + half_length, center[1] - half_width, center[2] + half_height]
    # ]
    kkk=1
    corner_lines = [[corners[0], corners[1]],
                    [corners[0], corners[2]],
                    [corners[0], corners[4]],
                    [corners[1], corners[3]],
                    [corners[1], corners[5]],
                    [corners[2], corners[3]],
                    [corners[2], corners[6]],
                    [corners[7], corners[5]],
                    [corners[7], corners[6]],
                    [corners[7], corners[3]],
                    [corners[4], corners[5]],
                    [corners[4], corners[6]]
                    ]

    return corner_lines

def rectangle_struct(
        center: ca.SX,
        size: ca.SX,
):
    min_point = center - size/2
    max_point = center + size/2
    corner_lines = compute_rectangle_edges(center, size)
    rect = {"max":max_point,
            "min":min_point,
            "corner_lines":corner_lines}
    return rect

def point_to_rectangle(
        p: ca.SX,
        rect: dict,
):
    dx = max(rect["min"][0] - p[0], 0, p[0] - rect["max"][0])
    dy = max(rect["min"][1] - p[1], 0, p[1] - rect["max"][1])
    dz = max(rect["min"][2] - p[2], 0, p[2] - rect["max"][2])
    return ca.sqrt(dx * dx + dy * dy + dz * dz)

def sphere_to_rectangle(
    sphere_center: ca.SX,
    rect: dict,
    sphere_radius: ca.SX,
) -> ca.SX:
    distance = point_to_rectangle(sphere_center, rect) - sphere_radius
    return distance

def capsule_to_rectangle(
        capsule_centers: List[ca.SX],
        rect: dict,
        capsule_radius: ca.SX
):
    min_distance = ca.inf
    for corner_line in rect["corner_lines"]:
        dist = line_to_line(corner_line[0], corner_line[1], capsule_centers[0], capsule_centers[1]) - capsule_radius
        min_distance = min(dist, min_distance)
    return min_distance
