from fabrics.helpers.geometric_primitives import (
    Sphere,
    Capsule,
)

def test_sphere_collision_link():
    sphere = Sphere('sphere_1', radius=0.1)
    assert sphere.radius == 0.1
    assert sphere.size == [0.1]


def test_capsule_collision_link():
    capsule = Capsule('capsule_1', radius=0.2, length=0.5)
    assert capsule.radius == 0.2
    assert capsule.length == 0.5
    assert capsule.size == [0.2, 0.5]
