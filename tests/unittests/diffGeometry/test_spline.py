import pytest
import casadi as ca
import numpy as np
from optFabrics.diffGeometry.splineTrajectory import SplineTrajectory


def test_ref_creation():
    t = ca.SX.sym("t")
    ctrlpts = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]
    J = ca.SX(np.identity(2))
    degree = 2
    n = 2
    splineTraj = SplineTrajectory(n, J, degree=degree, ctrlpts=ctrlpts, t=t)
    assert ca.is_equal(t, splineTraj.t())

@pytest.fixture
def simple_splineTrajectory():
    t = ca.SX.sym("t")
    ctrlpts = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]
    J = ca.SX(np.identity(2))
    degree = 2
    n = 2
    splineTraj = SplineTrajectory(n, J, degree=degree, ctrlpts=ctrlpts, t=t)
    return splineTraj

@pytest.mark.skip(reason="Spline trajectories currently not maintained")
def test_evaluate(simple_splineTrajectory):
    t = 0.5
    x, v, a = simple_splineTrajectory.evaluate(t)
    assert x[0] == pytest.approx(0.75)
    assert x[1] == pytest.approx(0.75)
    assert np.linalg.norm(v) == pytest.approx(1)

@pytest.mark.skip(reason="Spline trajectories currently not maintained")
def test_evaluateAtZero(simple_splineTrajectory):
    t = 0.0
    x, v, a = simple_splineTrajectory.evaluate(t)
    assert x[0] == pytest.approx(0.0)
    assert x[1] == pytest.approx(1.00)
