import pytest

from col_avoidance import pointMassAvoidance
#from pointMass import pointMass
from pointMass_dynamic import pointMassDynamicGoal
from nlink import nlink
from nlink_dynamic import nlinkDynamicGoal


def test_pointMassAvoidance():
    res = pointMassAvoidance(10)
    assert res is None

"""
def test_pointMass():
    res = pointMass(10)
    assert res
"""


def test_pointMassDynamic():
    res = pointMassDynamicGoal(n_steps=10)
    assert res is None


def test_nlink():
    res = nlink(n=3, n_steps=10)
    assert res is None


def test_nlinkDynamic():
    res = nlinkDynamicGoal(n=7, n_steps=10)
    assert res is None
