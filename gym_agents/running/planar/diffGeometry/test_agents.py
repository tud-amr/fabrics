import pytest

from col_avoidance_moving import pointMassDynamicAvoidance
from col_avoidance import pointMassAvoidance
from pointMass import pointMass
from pointMass_moving import pointMassDynamic
from pointMass_dynamic import pointMassDynamicGoal
from nlink import nlink
from nlink_moving import nlinkDynamic


def test_pointMassAvoidance():
    res = pointMassAvoidance(10)
    assert len(res['qs'][0]) == 10


def test_pointMassDynamicAvoidance():
    res = pointMassDynamicAvoidance(10)
    assert len(res['qs'][0]) == 10


def test_pointMass():
    res = pointMass(10)
    assert len(res['qs'][0]) == 10
    assert len(res['solverTimes'][0]) == 10


def test_pointMassDynamic():
    res = pointMassDynamic(10)
    assert len(res['qs'][0]) == 10
    assert len(res['qs'][0][0]) == 2
    assert len(res['solverTimes'][0]) == 10


def test_pointMassDynamicGoal():
    res = pointMassDynamicGoal(10)
    assert len(res['qs'][0]) == 10
    assert len(res['qs'][0][0]) == 2
    assert len(res['solverTimes'][0]) == 10


def test_nlink():
    res = nlink(n=3, n_steps=10)
    assert len(res['qs'][0]) == 10
    assert len(res['solverTimes'][0]) == 10
    assert len(res['qs'][0][0]) == 3
    res = nlink(n=1, n_steps=10)
    assert len(res['qs'][0]) == 10
    assert len(res['qs'][0][0]) == 1
    assert len(res['solverTimes'][0]) == 10


def test_nlinkDynamic():
    res = nlinkDynamic(n=7, n_steps=10)
    assert len(res['qs'][0]) == 10
    assert len(res['solverTimes'][0]) == 10
    assert len(res['qs'][0][0]) == 7
    res = nlinkDynamic(n=1, n_steps=10)
    assert len(res['qs'][0]) == 10
    assert len(res['qs'][0][0]) == 1
    assert len(res['solverTimes'][0]) == 10
