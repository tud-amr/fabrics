import casadi as ca
import numpy as np
from fabrics.helpers.geometric_primitives import (
    Sphere,
    Capsule,
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
