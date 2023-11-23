import casadi as ca
import numpy as np
from fabrics.helpers.geometric_primitives import (
    Cuboid,
    Sphere,
    Capsule,
    Plane,
)

def test_sphere():
    sphere = Sphere('sphere_1', radius=0.1)
    assert sphere.radius == 0.1
    assert sphere.size == [0.1]

def test_sphere_sphere_distance():
    sphere_1 = Sphere('sphere_1')
    sphere_2 = Sphere('sphere_1')
    sphere_1_position = ca.SX.sym("sphere_1_positon", 3)
    sphere_1.set_position(sphere_1_position)
    sphere_2.set_position(ca.SX(np.zeros(3)))
    distance = sphere_1.distance(sphere_2)
    assert isinstance(distance, ca.SX)
    fun = ca.Function('fun', [sphere_1_position, sphere_1.sym_radius, sphere_2.sym_radius], [distance])
    distance_value = fun(np.array([0.3, 0.2, 0.1]), 0.1, 0.2)
    distance_numpy = np.linalg.norm(np.array([0.3, 0.2, 0.1])) - 0.1 - 0.2
    assert distance_value == distance_numpy


def test_capsule():
    capsule = Capsule('capsule_1', radius=0.2, length=0.5)
    assert capsule.radius == 0.2
    assert capsule.length == 0.5
    assert capsule.size == [0.2, 0.5]


def test_cuboid():
    cuboid = Cuboid('cuboid', sizes=[0.3, 0.4, 0.1])
    assert cuboid.size[0] == 0.3
    assert cuboid.size[1] == 0.4
    assert cuboid.size[2] == 0.1
    assert str(cuboid) == 'Cuboid: cuboid'

def test_set_origin():
    capsule = Capsule('capsule', radius=1, length=0.5)
    origin = ca.SX(np.identity(4))
    origin[0:3, 3] = np.array([0.3, 0.2, 0.1])
    capsule.set_origin(origin)
    assert capsule.position[0] == 0.3
    centers = capsule.centers
    center_fun = ca.Function('center_fun', [capsule.sym_length], centers)
    centers_np = center_fun(np.array([0.5]))
    assert centers_np[0][0] == 0.3
    assert centers_np[0][1] == 0.2
    assert centers_np[0][2] == 0.35
    assert 'radius_capsule' in capsule.parameters

def test_plane():
    plane = Plane('plane', plane_equation = [0.0, 0.0, 1.0, 0.0])
    assert plane.size == [0.0, 0.0, 1.0, 0.0]
    assert plane.plane_equation == [0.0, 0.0, 1.0, 0.0]
    assert 'plane' in plane.sym_size
    assert isinstance(plane.sym_size, dict)
    assert ca.is_equal(plane.sym_plane_equation, plane.sym_size['plane'])
    assert 'plane' in plane.parameters




