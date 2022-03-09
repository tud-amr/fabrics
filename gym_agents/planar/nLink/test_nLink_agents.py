import pytest

from nlink import nlink
from nlink_dynamic import nlinkDynamicGoal
from nlink_moving import nlinkDynamicObst


def test_nlink():
    res = nlink(n=3, n_steps=10, render=False)
    assert isinstance(res, dict)

def test_nlinkDynamic():
    res = nlinkDynamicGoal(n=7, n_steps=10, render=False)
    assert isinstance(res, dict)


def test_nlinkDynamicObst():
    res = nlinkDynamicObst(n=7, n_steps=10, render=False)
    assert isinstance(res, dict)
