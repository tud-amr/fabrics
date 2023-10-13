import casadi as ca
import pytest
import numpy as np
from fabrics.helpers.variables import Variables

from fabrics.components.leaves.geometry import CapsuleCuboidLeaf, CapsuleSphereLeaf, LimitLeaf, PlaneConstraintGeometryLeaf, SelfCollisionLeaf


def test_limit_leaf():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    root_variables = Variables(
        state_variables={"q": q, "qdot": qdot}
    )
    limit_leaf_upper = LimitLeaf(root_variables, 0, 1.0, 1)
    assert ca.is_equal(limit_leaf_upper.map()._phi, 1.0 - q[0], 2)
    limit_leaf_lower = LimitLeaf(root_variables, 1, -1.0, 0)
    assert ca.is_equal(limit_leaf_lower.map()._phi, q[1] - -1.0, 2)

def test_self_collision_leaf():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    root_variables = Variables(
        state_variables={"q": q, "qdot": qdot}
    )
    fk = q[0] + q[1]
    self_collision_leaf = SelfCollisionLeaf(root_variables, fk, "test_self_collision")
    phi = ca.norm_2(fk) / (2 * 0.1) - 1
    phi_fun = ca.Function("ground_truth_phi", [q], [phi])
    phi_test = ca.Function("test_phi", [q], [self_collision_leaf.map()._phi])
    value = np.array([0.1, -0.334])
    phi_ground_truth = phi_fun(value)
    phi_test = phi_test(value)
    assert np.isclose(phi_ground_truth, phi_test)

def test_plane_constraint_leaf():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    root_variables = Variables(
        state_variables={"q": q, "qdot": qdot}
    )
    fk = ca.vcat([q[0] + q[1], 0, 0])
    plane_leaf = PlaneConstraintGeometryLeaf(root_variables, "x_plane", "link_0", fk)
    plane_leaf._map.concretize()
    value = np.array([0.3, 0.2])
    x, J, Jdot = plane_leaf._map.forward(
        q=value,
        qdot=np.zeros(2),
        radius_body_link_0=0.15,
        x_plane=np.array([-1.0, 0.0, 0.0, 1.0]),
    )
    assert x == pytest.approx(0.35)

def test_capsule_sphere_leaf():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    root_variables = Variables(
        state_variables={"q": q, "qdot": qdot}
    )
    fk_1 = ca.vcat([q[0] + q[1], 0.2, 0])
    fk_2 = ca.vcat([q[0] + q[1], -0.2, 0])
    capsule_sphere_leaf = CapsuleSphereLeaf(
        root_variables,
        "capsule",
        "sphere", 
        fk_1,
        fk_2,
    )
    capsule_sphere_leaf._map.concretize()
    value = np.array([0.2, 0.1])
    x, J, Jdot = capsule_sphere_leaf._map.forward(
        q=value,
        qdot=np.zeros(2),
        radius_capsule=0.15,
        x_sphere=np.array([0.7, 0.0, 0.0]),
        radius_sphere=0.2,
    )
    assert x == pytest.approx(0.05)

def test_capsule_cuboid_leaf():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    root_variables = Variables(
        state_variables={"q": q, "qdot": qdot}
    )
    fk_1 = ca.vcat([-q[0] - q[1], 0.1, 0])
    fk_2 = ca.vcat([-q[0] - q[1], -0.2, 0])
    capsule_cuboid_leaf = CapsuleCuboidLeaf(
        root_variables,
        "capsule",
        "cuboid", 
        fk_1,
        fk_2,
    )
    capsule_cuboid_leaf._map.concretize()
    value = np.array([0.2, 0.1])
    x, J, Jdot = capsule_cuboid_leaf._map.forward(
        q=value,
        qdot=np.zeros(2),
        radius_capsule=0.15,
        x_cuboid=np.array([0.7, 0.0, 0.0]),
        size_cuboid=np.array([0.2, 0.2, 0.2]),
    )
    assert x == pytest.approx(0.75)
