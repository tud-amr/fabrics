import pytest
import casadi as ca
import numpy as np
from fabrics.diffGeometry.geometry import Geometry
from fabrics.diffGeometry.energized_geometry import (
    EnergizedGeometry,
    WeightedGeometry,
)
from fabrics.diffGeometry.energy import Lagrangian, FinslerStructure
from fabrics.diffGeometry.diffMap import DifferentialMap
from fabrics.diffGeometry.variables import eps


@pytest.fixture
def energization_example():
    x = ca.SX.sym("x", 2)
    xdot = ca.SX.sym("xdot", 2)
    h = 0.5 / (x ** 2) * ca.norm_2(xdot) ** 2
    geo = Geometry(h=h, x=x, xdot=xdot)
    l = 1.0 * ca.norm_2(xdot)
    le = FinslerStructure(l, x=x, xdot=xdot)
    return geo, le


@pytest.fixture
def energization_example_pulled():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    x = ca.SX.sym("x", 2)
    xdot = ca.SX.sym("xdot", 2)
    phi = ca.vertcat(ca.cos(q[1]) * q[0], ca.sin(q[1]) * q[0])
    dm = DifferentialMap(phi, q=q, qdot=qdot, Jdot_sign=+1)
    h = 0.5 * ca.norm_2(xdot) ** 2 / (x ** 2 + eps)
    geo = Geometry(h=h, x=x, xdot=xdot)
    l = 0.5 * ca.dot(x, x) * ca.dot(xdot, xdot)
    le = Lagrangian(l, var=[x, xdot])
    return geo, le, dm


@pytest.fixture
def two_energizations():
    x = ca.SX.sym("x", 2)
    xdot = ca.SX.sym("xdot", 2)
    h = 0.5 * ca.norm_2(xdot) ** 2 / (x ** 2 + eps)
    geo = Geometry(h=h, x=x, xdot=xdot)
    l1 = 0.5 * ca.dot(x, x) * ca.dot(xdot, xdot)
    le1 = Lagrangian(l1, var=[x, xdot])
    l2 = 2.5 * ca.dot(xdot, xdot) / ca.dot(x, x)
    le2 = Lagrangian(l2, x=x, xdot=xdot)
    return geo, le1, le2


@pytest.fixture
def two_different_spaces():
    q = ca.SX.sym("q", 2)
    qdot = ca.SX.sym("qdot", 2)
    # polar coordinates
    x1 = ca.SX.sym("x", 2)
    xdot1 = ca.SX.sym("xdot", 2)
    phi1 = ca.vertcat(ca.cos(q[1]) * q[0], ca.sin(q[1]) * q[0])
    dm1 = DifferentialMap(phi1, q=q, qdot=qdot, Jdot_sign=+1)
    h1 = 0.5 * ca.norm_2(xdot1) ** 2 / (x1 ** 2 + eps)
    geo1 = Geometry(h=h1, x=x1, xdot=xdot1)
    l1 = 0.5 * ca.dot(x1, x1) * ca.dot(xdot1, xdot1)
    le1 = Lagrangian(l1, x=x1, xdot=xdot1)
    # space of q[1]
    x2 = ca.SX.sym("x", 1)
    xdot2 = ca.SX.sym("xdot", 1)
    phi2 = q[1]
    dm2 = DifferentialMap(phi2, q=q, qdot=qdot, Jdot_sign=+1)
    h2 = 0.5 * ca.norm_2(xdot2) ** 2 / (x2 ** 2 + eps)
    geo2 = Geometry(h=h2, x=x2, xdot=xdot2)
    l2 = 0.5 * ca.dot(x2, x2) * ca.dot(xdot2, xdot2)
    le2 = Lagrangian(l2, x=x2, xdot=xdot2)
    return geo1, le1, dm1, geo2, le2, dm2


def test_energization_simple(energization_example):
    geo = energization_example[0]
    le = energization_example[1]
    geo_weighted = WeightedGeometry(g=geo, le=le)
    geo_weighted.concretize()
    x = np.array([0.2, -0.8])
    xdot = np.array([-0.5, -1.4])
    xdot_norm2 = xdot[0] ** 2 + xdot[1] ** 2
    pe = np.array(
        [
            [1 - ((xdot[0] ** 2) / xdot_norm2), -xdot[0] * xdot[1] / xdot_norm2],
            [-xdot[0] * xdot[1] / xdot_norm2, 1 - (xdot[1] ** 2 / xdot_norm2)],
        ]
    )
    h = 0.5 / (x ** 2) * np.linalg.norm(xdot) ** 2
    f_test = np.dot(pe, h)
    M_w, f_w, xddot_w, alpha_ex = geo_weighted.evaluate(x, xdot)
    xddot = xddot_w - alpha_ex * xdot
    h_alpha = h + alpha_ex * xdot
    assert M_w[0, 0] == 1.0
    assert M_w[0, 1] == 0.0
    assert M_w[1, 0] == 0.0
    assert M_w[1, 1] == 1.0
    assert xddot[0] == pytest.approx(-f_test[0])
    assert xddot[1] == pytest.approx(-f_test[1])
    assert h_alpha[0] == pytest.approx(f_test[0])
    assert h_alpha[1] == pytest.approx(f_test[1])


def test_pull_energized(energization_example_pulled):
    geo, le, dm = energization_example_pulled
    geo.concretize()
    dm.concretize()
    geo_energized = EnergizedGeometry(geo, le)
    geo_energized.concretize()
    geo_weighted = WeightedGeometry(g=geo, le=le)
    geo_weighted.concretize()
    q = np.array([0.5, np.pi / 4])
    qdot = np.array([-0.0, 5.2])
    x, J, Jdot = dm.forward(q, qdot)
    Jt = np.transpose(J)
    xdot = np.dot(J, qdot)
    h_0, xddot_0 = geo.evaluate(x, xdot)
    M, f, xddot = geo_energized.evaluate(x, xdot)
    M_w, f_w, xddot_w, alpha_w = geo_weighted.evaluate(x, xdot)
    xddot_w_alpha = xddot_w - alpha_w * xdot
    h = 0.5 * np.linalg.norm(xdot) ** 2 / (x ** 2 + eps)
    xddot_w_test = -np.dot(np.dot(np.linalg.pinv(M_w), M_w), h)
    f_w_test = np.dot(M_w, h)
    assert M_w[0, 0] == pytest.approx(x[0] ** 2 + x[1] ** 2)
    assert M_w[0, 1] == pytest.approx(0.0)
    assert M_w[1, 0] == pytest.approx(0.0)
    assert M_w[1, 1] == pytest.approx(x[0] ** 2 + x[1] ** 2)
    assert M_w[0, 0] == pytest.approx(x[0] ** 2 + x[1] ** 2)
    assert M_w[0, 1] == pytest.approx(0.0)
    assert M_w[1, 0] == pytest.approx(0.0)
    assert M_w[1, 1] == pytest.approx(x[0] ** 2 + x[1] ** 2)
    assert f_w[0] == pytest.approx(f_w_test[0])
    assert f_w[1] == pytest.approx(f_w_test[1])
    assert xddot_w[0] == pytest.approx(xddot_w_test[0])
    assert xddot_w[1] == pytest.approx(xddot_w_test[1])
    geo_pulled = geo.pull(dm)
    geo_pulled.concretize()
    geo_energized_pulled = geo_energized.pull(dm)
    geo_energized_pulled.concretize()
    geo_weighted_pulled = geo_weighted.pull(dm)
    geo_weighted_pulled.concretize()
    h_p_0, qddot_0 = geo_pulled.evaluate(q, qdot)
    M_p, f_p, qddot = geo_energized_pulled.evaluate(q, qdot)
    M_p_w, f_p_w, qddot_w, alpha_p_w = geo_weighted_pulled.evaluate(q, qdot)
    qddot_w_alpha = qddot_w - alpha_p_w * qdot
    f_p_w_test = np.dot(Jt, np.dot(M_w, h)) + np.dot(
        Jt, np.dot(M_w, np.dot(Jdot, qdot))
    )
    M_p_test = np.dot(np.transpose(J), np.dot(M, J))
    assert M_p[0, 0] == pytest.approx(M_p_test[0, 0])
    assert M_p[0, 1] == pytest.approx(M_p_test[0, 1])
    assert M_p[1, 0] == pytest.approx(M_p_test[1, 0])
    assert M_p[1, 1] == pytest.approx(M_p_test[1, 1])
    assert M_p_w[0, 0] == pytest.approx(M_p_test[0, 0])
    assert M_p_w[0, 1] == pytest.approx(M_p_test[0, 1])
    assert M_p_w[1, 0] == pytest.approx(M_p_test[1, 0])
    assert M_p_w[1, 1] == pytest.approx(M_p_test[1, 1])
    assert f_p_w[0] == pytest.approx(f_p_w_test[0])
    assert f_p_w[1] == pytest.approx(f_p_w_test[1])
    assert qddot[0] == pytest.approx(qddot_w_alpha[0], rel=1e-4)
    assert qddot[1] == pytest.approx(qddot_w_alpha[1], rel=1e-4)
    xddot_p = np.dot(J, qddot) + np.dot(Jdot, qdot)
    assert xddot_p[0] == pytest.approx(xddot[0], rel=1e-4)
    assert xddot_p[1] == pytest.approx(xddot[1], rel=1e-4)


def test_sum_energization(two_energizations):
    geo, le1, le2 = two_energizations
    en_1 = EnergizedGeometry(geo, le1)
    en_2 = EnergizedGeometry(geo, le2)
    we_1 = WeightedGeometry(g=geo, le=le1)
    we_2 = WeightedGeometry(g=geo, le=le2)
    en = en_1 + en_2
    we = we_1 + we_2
    en.concretize()
    we.concretize()
    x = np.array([0.2, -0.8])
    xdot = np.array([-0.5, -1.4])
    M_en, f_en, xddot_en = en.evaluate(x, xdot)
    M_we, f_we, xddot_we, alpha_we = we.evaluate(x, xdot)
    xddot_we_alpha = xddot_we - alpha_we * xdot
    assert M_en[0, 0] == pytest.approx(M_we[0, 0])
    assert M_en[0, 1] == pytest.approx(M_we[0, 1])
    assert M_en[1, 0] == pytest.approx(M_we[1, 0])
    assert M_en[1, 1] == pytest.approx(M_we[1, 1])
    assert xddot_we_alpha[0] == pytest.approx(xddot_en[0])
    assert xddot_we_alpha[1] == pytest.approx(xddot_en[1])


def test_sum_energization_man_compute_rhs(two_energizations):
    geo, le1, le2 = two_energizations
    en_1 = EnergizedGeometry(geo, le1)
    en_2 = EnergizedGeometry(geo, le2)
    we_1 = WeightedGeometry(g=geo, le=le1)
    we_2 = WeightedGeometry(g=geo, le=le2)
    en_1.concretize()
    en_2.concretize()
    we_1.concretize()
    we_2.concretize()
    en = en_1 + en_2
    we = we_1 + we_2
    en.concretize()
    we.concretize()
    x = np.array([0.2, -0.8])
    xdot = np.array([-0.5, -1.4])
    M_en1, f_en1, xddot_en1 = en_1.evaluate(x, xdot)
    M_en2, f_en2, xddot_en2 = en_2.evaluate(x, xdot)
    M_en, f_en, xddot_en = en.evaluate(x, xdot)
    M_we1, f_we1, xddot_we1, alpha_we1 = we_1.evaluate(x, xdot)
    M_we2, f_we2, xddot_we2, alpha_we2 = we_2.evaluate(x, xdot)
    M_we, f_we, xddot_we, alpha_we = we.evaluate(x, xdot)
    f_we1_we2_alpha = (
        f_we1
        + np.dot(M_we1, alpha_we1 * xdot)
        + f_we2
        + np.dot(M_we2, alpha_we2 * xdot)
    )
    assert f_we1_we2_alpha[0] == pytest.approx(f_en[0])
    assert f_we1_we2_alpha[1] == pytest.approx(f_en[1])


def test_two_spaces_energization(two_different_spaces):
    geo1, le1, dm1, geo2, le2, dm2 = two_different_spaces
    en_1 = EnergizedGeometry(geo1, le1).pull(dm1)
    en_2 = EnergizedGeometry(geo2, le2).pull(dm2)
    we_1 = WeightedGeometry(g=geo1, le=le1).pull(dm1)
    we_2 = WeightedGeometry(g=geo2, le=le2).pull(dm2)
    en_1.concretize()
    en_2.concretize()
    we_1.concretize()
    we_2.concretize()
    en = en_1 + en_2
    we = we_1 + we_2
    en.concretize()
    we.concretize()
    q = np.array([0.2, -0.8])
    qdot = np.array([-1.1, 0.6])
    M_en1, f_en1, qddot_en1 = en_1.evaluate(q, qdot)
    M_en2, f_en2, qddot_en2 = en_2.evaluate(q, qdot)
    M_en, f_en, qddot_en = en.evaluate(q, qdot)
    M_we1, f_we1, qddot_we1, alpha_we1 = we_1.evaluate(q, qdot)
    M_we2, f_we2, qddot_we2, alpha_we2 = we_2.evaluate(q, qdot)
    M_we, f_we, qddot_we, alpha_we = we.evaluate(q, qdot)
    f_we1_we2_alpha = (
        f_we1
        + np.dot(M_we1, alpha_we1 * qdot)
        + f_we2
        + np.dot(M_we2, alpha_we2 * qdot)
    )
    f_test = f_we1 + f_we2 + np.dot(M_we, alpha_we * qdot)
    assert f_we1_we2_alpha[0] == pytest.approx(f_en[0])
    assert f_we1_we2_alpha[1] == pytest.approx(f_en[1])
    assert M_en[0, 0] == pytest.approx(M_we[0, 0])
    assert M_en[0, 1] == pytest.approx(M_we[0, 1])
    assert M_en[1, 0] == pytest.approx(M_we[1, 0])
    assert M_en[1, 1] == pytest.approx(M_we[1, 1])
    qddot_we_alpha = qddot_we - alpha_we * qdot
    qddot_we12_alpha = -np.dot(np.linalg.pinv(M_we), f_we1_we2_alpha)
    assert qddot_we12_alpha[0] == pytest.approx(qddot_en[0], rel=1e-4)
    assert qddot_we12_alpha[1] == pytest.approx(qddot_en[1], rel=1e-4)
    # !!! Important individual energization is different from combined energization
    # only the latter is what we need
    assert qddot_we_alpha[0] != pytest.approx(qddot_en[0])
    assert qddot_we_alpha[1] != pytest.approx(qddot_en[1])
