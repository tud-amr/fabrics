from fabrics.components.robot_representation.collision_link import (
    Sphere,
    Capsule,
)

def test_sphere_collision_link():
    sphere = Sphere(radius=0.1)
    assert sphere.radius == 0.1
    assert sphere.size == [0.1]


def test_capsule_collision_link():
    capsule = Capsule(radius=0.2, length=0.5)
    assert capsule.radius == 0.2
    assert capsule.length == 0.5
    assert capsule.size == [0.2, 0.5]
