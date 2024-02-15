import numpy as np
import casadi as ca

def rectangle_center_size_to_struct(center, size):
    min_point = center - size/2
    max_point = center + size/2
    plane_list = compute_rectangle_planes(center, size)
    rect = {"max":max_point,
            "min":min_point,
            "planes":plane_list}
    return rect

def compute_rectangle_planes(center, size):
    half_length = size[0] / 2
    half_width = size[1] / 2
    half_height = size[2] / 2

    # Define the equations for the six planes of the rectangle
    planes = [
        # Front plane
        [1, 0, 0, center[0] + half_length],
        # Back plane
        [-1, 0, 0, -(center[0] - half_length)],
        # Top plane
        [0, 1, 0, center[1] + half_width],
        # Bottom plane
        [0, -1, 0, -(center[1] - half_width)],
        # Left plane
        [0, 0, 1, center[2] + half_height],
        # Right plane
        [0, 0, -1, -(center[2] - half_height)]
    ]
    return planes

def construct_plane(edge_points):
    """
    Construct plane based on edge points
    """
    p1 = edge_points[0]
    p2 = edge_points[1]
    p3 = edge_points[2]
    normal = ca.cross((p2 - p1), (p3 - p2))
    center = (p3-p1)/2 + p1
    plane = np.append(normal, center[2])
    return plane

def distance_rect_to_point(rect, p):
    """
    rect must be a dict with a minimum and maximum point of the rectangle in xyz
    p must be a vector of xyz
    returns the minimum distance between a rectangle and a point in 3D space
    """
    dx = max(rect["min"][0] - p[0], 0, p[0] - rect["max"][0])
    dy = max(rect["min"][1] - p[1], 0, p[1] - rect["max"][1])
    dz = max(rect["min"][2] - p[2], 0, p[2] - rect["max"][2])
    return np.sqrt(dx*dx + dy*dy + dz*dz)

def distance_rect_to_sphere(rect, center_sphere, radius_sphere):
    dist_to_center = distance_rect_to_point(rect, center_sphere)
    dist = dist_to_center - radius_sphere
    return dist

# def distance_rect_to_capsule(rect, capsule_centers, capsule_radius):
#     """
#     Use line to plane formulation
#     """
#     min_dist = 10e6
#     for plane in range(rect["planes"]):
#         dist = capsule_to_plane(capsule_centers, plane, capsule_radius)
#         min_dist = min(min_dist, dist)

def check_dist_rect_point(p, dist):
    dist_hardcode = 0.0
    dist_p1 = p - np.array([1.5, 0.5, 0.0])
    dist_p2 = p - np.array([0.5, 0.0, 1.0])
    all_zeros1 = not np.any(dist_p1)
    all_zeros2 = not np.any(dist_p2)
    if all_zeros1:
        dist_hardcode = 0.5
        assert dist_hardcode == dist
    elif all_zeros2:
        dist_hardcode = 0.0
        assert dist_hardcode == dist
    else:
        print("warning, test 'rectangle to point' failed")

# rect = {"max":{"x":1.0, "y":1.0, "z":1.0}, "min":{"x":0.0, "y":0.0, "z":0.0}}
center_box = np.array([0.5, 0.5, 0.5])
size_box = np.array([1, 1, 1])
rect = rectangle_center_size_to_struct(center=center_box, size=size_box)
p_list = [np.array([1.5, 0.5, 0.0]), np.array([0.5, 0.0, 1.0])]
for p in p_list:
    dist = distance_rect_to_point(rect, p)
    check_dist_rect_point(p, dist)
    print("rect:", rect)
    print("distance:", dist)

capsule_centers = [np.array([-1.0, 0.0, 0.0]), np.array([-1.0, -1.0, 0.0])]
capsule_radius = [0.0, 0.0]
# for p in p_list:
#     distance_rect_to_capsule(rect, capsule_centers, capsule_radius)




